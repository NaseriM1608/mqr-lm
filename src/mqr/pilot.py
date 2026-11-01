"""Fresh, bounded validation pilot for head initialization and early learning.

This driver deliberately has no resume option. Its short learning-rate schedule
is a new experiment, independent of the original 3B-token calibration schedule.
Timing recorded here is diagnostic; parallel seed processes cannot compare speed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .config import ExperimentConfig, HeadConfig, ModelConfig, TrainConfig
from .data import TokenStream, load_tokenizer
from .kaggle import inspect_device
from .model import LanguageModel
from .provenance import file_sha256, prepared_identity
from .training import autocast_context, resolve_device, save_checkpoint, set_seed
from .tuning import optimizer_attempt, window_digest


LANGUAGES = ("en", "de", "ar", "fa")
ARMS = ("dense", "direct1", "direct4", "mqr", "direct1_scale", "direct4_scale", "mqr_scale")
EVALUATION_STEPS = (0, 16, 32, 64, 128)
PROTOCOL_VERSION = "mqr-learning-pilot-v1"


def _json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def pilot_configs(*, vocab_size: int, seed: int = 0, microbatch: int = 1,
                  device: str = "cuda:0", cpu_test: bool = False) -> dict[str, ExperimentConfig]:
    """Resolve all seven arms; scale variants have the same canonical config."""
    if type(seed) is not int or seed not in (0, 1):
        raise ValueError("The registered pilot uses seeds 0 and 1.")
    if type(microbatch) is not int or microbatch < 1 or 32 % microbatch:
        raise ValueError("microbatch must divide the fixed global batch of 32.")
    if cpu_test and device != "cpu":
        raise ValueError("cpu_test requires device=cpu.")
    model = (ModelConfig(vocab_size=vocab_size, layers=1, d_model=16, attention_heads=2,
                         ffn_dim=32, context_length=8)
             if cpu_test else ModelConfig(vocab_size=vocab_size, layers=12, d_model=512,
                                          attention_heads=8, ffn_dim=2048, context_length=512))
    training = TrainConfig(seed=seed, device=device, precision="fp32" if device == "cpu" else "fp16",
                           batch_size=microbatch, gradient_accumulation=32 // microbatch,
                           max_steps=256, learning_rate=3e-4, warmup_steps=16,
                           min_lr_ratio=0.1, checkpoint_every=0, eval_every=0,
                           loss_chunk_positions=64, num_threads=2)
    base = HeadConfig(d_e=8 if cpu_test else 128, d_c=12 if cpu_test else 256, num_queries=4)
    heads = {
        "dense": replace(base, kind="dense", d_e=model.d_model, num_queries=1),
        "direct1": replace(base, kind="direct", num_queries=1),
        "direct4": replace(base, kind="direct", num_queries=4),
        "mqr": replace(base, kind="mqr"),
    }
    return {arm: ExperimentConfig(model, heads[arm.removesuffix("_scale")], training) for arm in ARMS}


def build_window_bank(tokens: np.ndarray, manifest: dict, *, split: str, context_length: int,
                      windows_per_language: int, seed: int) -> dict:
    """Sample disjoint (context+1)-token blocks within each language span.

    Banks are shared across model seeds and microbatch choices. Hashes cover
    offsets and canonical uint32 token content, including each final target.
    """
    if split not in ("train", "validation"):
        raise ValueError("Pilot banks may use training or validation data only.")
    if type(context_length) is not int or context_length < 1 or type(windows_per_language) is not int or windows_per_language < 1:
        raise ValueError("Context length and windows per language must be positive integers.")
    if tokens.ndim != 1 or tokens.dtype.kind not in "ui":
        raise ValueError("Bank tokens must be a one-dimensional unsigned/integer array.")
    try:
        slices = manifest["splits"][split]["languages"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Prepared manifest must declare per-language token spans.") from exc
    generator = torch.Generator(device="cpu").manual_seed(seed)
    spans, records = [], []
    content = hashlib.sha256()
    width = context_length + 1
    for language in LANGUAGES:
        try:
            start, end = slices[language]["token_start"], slices[language]["token_end"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Missing {split} token span for {language}.") from exc
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(tokens):
            raise ValueError(f"Invalid {split} token span for {language}.")
        if any(start < old_end and end > old_start for old_start, old_end in spans):
            raise ValueError("Language token spans overlap.")
        spans.append((start, end))
        blocks = (end - start) // width
        if blocks < windows_per_language:
            raise ValueError(f"Insufficient disjoint {split} windows for {language}.")
        selected = sorted(torch.randperm(blocks, generator=generator)[:windows_per_language].tolist())
        for block in selected:
            offset = start + block * width
            raw = np.asarray(tokens[offset:offset + width], dtype="<u4").tobytes()
            content.update(raw)
            records.append({"language": language, "token_start": offset, "token_end": offset + width,
                            "token_sha256": hashlib.sha256(raw).hexdigest()})
    bank = {"format_version": 1, "split": split, "context_length": context_length,
            "windows_per_language": windows_per_language, "languages": list(LANGUAGES),
            "sampling_seed": seed, "nonoverlap_scope": "all input and target tokens within each bank",
            "predicted_tokens": len(records) * context_length, "records": records,
            "content_sha256": content.hexdigest()}
    return {**bank, "bank_sha256": _json_digest(bank)}


def bank_tensors(tokens: np.ndarray, bank: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Recheck a frozen bank before materializing its small CPU tensors."""
    descriptor = {key: value for key, value in bank.items() if key != "bank_sha256"}
    if bank.get("bank_sha256") != _json_digest(descriptor):
        raise ValueError("Window bank descriptor hash differs.")
    arrays, content = [], hashlib.sha256()
    for record in bank["records"]:
        values = np.asarray(tokens[record["token_start"]:record["token_end"]], dtype="<u4")
        if len(values) != bank["context_length"] + 1:
            raise ValueError("Window bank extends outside its token stream.")
        raw = values.tobytes()
        if hashlib.sha256(raw).hexdigest() != record["token_sha256"]:
            raise ValueError("Window bank token content differs.")
        content.update(raw)
        arrays.append(values.astype(np.int64))
    if content.hexdigest() != bank["content_sha256"]:
        raise ValueError("Window bank aggregate token content differs.")
    values = np.stack(arrays)
    return torch.from_numpy(values[:, :-1].copy()), torch.from_numpy(values[:, 1:].copy())


@torch.inference_mode()
def evaluate_bank(model: LanguageModel, tokens: np.ndarray, bank: dict, *, microbatch: int = 1,
                  precision: str = "fp32", chunk_positions: int = 64) -> dict:
    """Evaluate exactly the stored windows, independent of training microbatch."""
    if type(microbatch) is not int or microbatch < 1 or chunk_positions < 1:
        raise ValueError("Evaluation microbatch and position chunks must be positive.")
    x, y = bank_tensors(tokens, bank)
    device, was_training = next(model.parameters()).device, model.training
    totals = {language: {"nll_sum": 0.0, "correct": 0, "predicted_tokens": 0, "window_nll": []}
              for language in LANGUAGES}
    model.eval()
    try:
        for offset in range(0, len(x), microbatch):
            bx, by = x[offset:offset + microbatch].to(device), y[offset:offset + microbatch].to(device)
            with autocast_context(device, precision):
                features, _ = model.backbone(bx)
                for local in range(len(bx)):
                    nll_sum, correct = 0.0, 0
                    for position in range(0, by.size(1), chunk_positions):
                        targets = by[local, position:position + chunk_positions]
                        logits = model.head.score_all(features[local, position:position + chunk_positions]).float()
                        nll_sum += F.cross_entropy(logits, targets, reduction="sum").item()
                        correct += (logits.argmax(-1) == targets).sum().item()
                    if not math.isfinite(nll_sum):
                        raise FloatingPointError("Nonfinite validation NLL.")
                    language = bank["records"][offset + local]["language"]
                    total = totals[language]
                    total["nll_sum"] += nll_sum
                    total["correct"] += correct
                    total["predicted_tokens"] += by.size(1)
                    total["window_nll"].append(nll_sum / by.size(1))
    finally:
        model.train(was_training)
    by_language = {}
    for language, total in totals.items():
        count = total["predicted_tokens"]
        nll = total["nll_sum"] / count
        by_language[language] = {**total, "full_nll": nll, "accuracy": total["correct"] / count}
    count = sum(total["predicted_tokens"] for total in totals.values())
    nll = sum(total["nll_sum"] for total in totals.values()) / count
    return {"bank_sha256": bank["bank_sha256"], "predicted_tokens": count, "full_nll": nll,
            "macro_nll": sum(value["full_nll"] for value in by_language.values()) / len(LANGUAGES),
            "per_language": by_language}


@torch.no_grad()
def probe_features(model: LanguageModel, tokens: np.ndarray, bank: dict,
                   *, positions_per_window: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract FP32 features only from the locked training-only probe."""
    if bank["split"] != "train":
        raise ValueError("Initialization probes must use training data.")
    x, y = bank_tensors(tokens, bank)
    positions = torch.linspace(0, bank["context_length"] - 1,
                               min(positions_per_window, bank["context_length"])).long()
    device, was_training = next(model.parameters()).device, model.training
    features, targets = [], []
    model.eval()
    try:
        for bx, by in zip(x, y):
            output, _ = model.backbone(bx[None].to(device))
            features.append(output[0, positions.to(device)].float())
            targets.append(by[positions].to(device))
    finally:
        model.train(was_training)
    return torch.cat(features), torch.cat(targets)


def _tensor_state_digest(values: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def finite_state_audit(model, optimizer, scaler=None) -> dict:
    """Audit live state for every arm, including trials without saved weights."""
    counts = {"model_tensors": 0, "optimizer_tensors": 0, "scaler_tensors": 0}
    failures = []

    def visit(value, label, path):
        if isinstance(value, torch.Tensor):
            counts[label] += 1
            if value.is_floating_point() and not torch.isfinite(value).all().item():
                failures.append(path)
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child, label, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, label, f"{path}.{index}")
        elif isinstance(value, float) and not math.isfinite(value):
            failures.append(path)

    visit(model.state_dict(), "model_tensors", "model")
    visit(optimizer.state_dict(), "optimizer_tensors", "optimizer")
    visit(scaler.state_dict() if scaler is not None else {}, "scaler_tensors", "scaler")
    result = {"finite": not failures, "nonfinite_paths": failures, **counts}
    if failures:
        raise FloatingPointError(f"Final state contains nonfinite values: {failures[:3]}")
    return result


def retain_checkpoint(arm: str, seed: int) -> bool:
    """Five full optimizer checkpoints across the complete two-seed study."""
    return arm in ("mqr", "mqr_scale") or (arm == "dense" and seed == 0)


def audit_checkpoint(path: str | Path, config: ExperimentConfig, *, completed_steps: int,
                     predicted_tokens: int, data_identity: dict) -> dict:
    """Reload final weights and optimizer state on CPU without constructing a model."""
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"format_version", "config", "model", "optimizer", "completed_steps", "predicted_tokens",
                "sampler_rng", "rng", "schedule", "scaler", "train_fingerprint", "data_identity"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("Final checkpoint is missing required training state.")
    if payload["format_version"] != 1 or ExperimentConfig.from_dict(payload["config"]) != config:
        raise ValueError("Final checkpoint configuration differs.")
    if (payload["completed_steps"] != completed_steps or payload["predicted_tokens"] != predicted_tokens
            or payload["data_identity"] != data_identity
            or payload["schedule"] != {"completed_steps": completed_steps, "max_steps": config.training.max_steps}):
        raise ValueError("Reloaded checkpoint counters, schedule or data identity differ.")
    checked = {"model": 0, "optimizer": 0, "scaler": 0}

    def visit(value, group):
        if isinstance(value, torch.Tensor):
            checked[group] += 1
            if value.device.type != "cpu":
                raise ValueError("Checkpoint did not reload entirely onto CPU.")
            if value.is_floating_point() and not torch.isfinite(value).all().item():
                raise FloatingPointError(f"Reloaded checkpoint has nonfinite {group} values.")
        elif isinstance(value, dict):
            for child in value.values():
                visit(child, group)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child, group)
        elif isinstance(value, float) and not math.isfinite(value):
            raise FloatingPointError(f"Reloaded checkpoint has nonfinite {group} values.")

    for group in checked:
        visit(payload[group], group)
    if not checked["model"] or not checked["optimizer"]:
        raise ValueError("Final checkpoint has no model or optimizer tensors.")
    if config.training.precision == "fp16" and not payload["scaler"]:
        raise ValueError("Final FP16 checkpoint lacks scaler state.")
    if payload["sampler_rng"].device.type != "cpu" or not isinstance(payload["rng"], dict):
        raise ValueError("Final checkpoint lacks portable RNG state.")
    optimizer_steps = [int(state["step"].item()) for state in payload["optimizer"]["state"].values()]
    if any(step != completed_steps for step in optimizer_steps):
        raise ValueError("Optimizer state counters differ from successful update count.")
    return {"cpu_reload": True, "finite_model_optimizer_scaler": True,
            "checked_tensor_counts": checked, "optimizer_parameter_states": len(optimizer_steps),
            "completed_steps": completed_steps, "predicted_tokens": predicted_tokens,
            "sha256": file_sha256(path), "bytes": path.stat().st_size}


def _gradient_groups(model) -> dict:
    groups = {}
    for name, parameter in model.named_parameters():
        group = "backbone" if name.startswith("backbone.") else ".".join(name.split(".")[:2])
        if parameter.grad is not None:
            groups.setdefault(group, {"squared_norm": 0.0, "elements": 0})
            groups[group]["squared_norm"] += parameter.grad.float().square().sum().item()
            groups[group]["elements"] += parameter.numel()
    return {name: {"unscaled_preclip_norm": math.sqrt(value["squared_norm"]),
                   "unscaled_preclip_rms": math.sqrt(value["squared_norm"] / value["elements"])}
            for name, value in groups.items()}


def _cpu_optimizer(optimizer) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.cpu()


def _sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_arm(config: ExperimentConfig, arm: str, train: TokenStream, validation_tokens: np.ndarray,
             train_bank: dict, validation_bank: dict, output: Path, *, stop_after: int,
             identity: dict, dense_target_std: float | None, stage: str) -> dict:
    # Imported lazily so dry-run planning remains independent of GPU diagnostics.
    from .head_diagnostics import diagnose_head, calibrate_initial_query_gain

    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "config.json", config.to_dict())
    tc = config.training
    torch.set_num_threads(tc.num_threads)
    set_seed(tc.seed)
    device = resolve_device(tc.device)
    model = LanguageModel(config).to(device).train()
    initial_backbone_sha = _tensor_state_digest(model.backbone.state_dict())
    features, targets = probe_features(model, train.tokens, train_bank)
    initial = diagnose_head(model.head, features, targets)
    transform = {"kind": "canonical_initialization", "applied": False, "gain": 1.0,
                 "training_probe_bank_sha256": train_bank["bank_sha256"]}
    if arm.endswith("_scale"):
        if dense_target_std is None:
            raise ValueError("Dense initialization target must be measured first.")
        calibration = calibrate_initial_query_gain(model.head, features, target_logit_std=dense_target_std)
        transform = {**calibration, "gain": calibration["query_projection_gain"],
                     "training_probe_bank_sha256": train_bank["bank_sha256"],
                     "applied": True, "kind": "experimental_initial_query_projection_gain"}
    after = diagnose_head(model.head, features, targets) if transform["applied"] else initial
    initialization = {"canonical_config": config.to_dict(), "config_sha256": _json_digest(config.to_dict()),
                      "initial_backbone_sha256": initial_backbone_sha, "transform": transform,
                      "before": initial, "after": after,
                      "note": "Initialization experiment only; raw-dot-product forward architecture is unchanged."}
    _write_json(output / "initialization.json", initialization)
    del features, targets
    model.zero_grad(set_to_none=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc.learning_rate, betas=tc.betas,
                                  weight_decay=tc.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if tc.precision == "fp16" else None
    generator = torch.Generator(device="cpu").manual_seed(tc.seed + 101)
    completed, attempted_tokens, successful_tokens, overflows = 0, 0, 0, 0
    consecutive_overflows = 0
    successful_window_digests = []
    evaluations = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    milestones = set(EVALUATION_STEPS) | {stop_after}
    with (output / "metrics.jsonl").open("x", encoding="utf-8") as events:
        def emit(value):
            events.write(json.dumps(value, allow_nan=False) + "\n")
            events.flush()

        def evaluate_now():
            result = evaluate_bank(model, validation_tokens, validation_bank, microbatch=1,
                                   precision=tc.precision, chunk_positions=tc.loss_chunk_positions)
            evaluations[str(completed)] = result
            emit({"event": "validation", "step": completed, **result})
            print(f"pilot seed={tc.seed} arm={arm} step={completed} macro_nll={result['macro_nll']:.6f}", flush=True)

        if stage == "learn":
            evaluate_now()
            pending_batch = None
            while completed < stop_after:
                _sync(device)
                started = time.perf_counter()
                if pending_batch is None:
                    pending_batch = train.sample(32, generator)
                x, y = pending_batch
                digest = window_digest(x, y)
                gradient_groups = {}
                milestone = completed + 1 in milestones or completed == 0

                def capture_groups(module):
                    gradient_groups.update(_gradient_groups(module))

                result = optimizer_attempt(model, optimizer, scaler, config, x, y, completed,
                                           before_clip=capture_groups if milestone else None)
                _sync(device)
                elapsed = time.perf_counter() - started
                attempted_tokens += result["attempted_tokens"]
                if result["skipped"]:
                    overflows += 1
                    consecutive_overflows += 1
                    # Retry this same batch; every arm must learn from the same
                    # successful windows even if GradScaler needs to recover.
                    emit({"event": "amp_overflow", "step": completed, "window_sha256": digest,
                          "attempt_seconds": elapsed, **result})
                    if consecutive_overflows >= 20 or overflows >= 40:
                        raise FloatingPointError("Pilot exceeded its bounded AMP retry budget.")
                    continue
                consecutive_overflows = 0
                completed += 1
                successful_tokens += result["attempted_tokens"]
                successful_window_digests.append(digest)
                pending_batch = None
                emit({"event": "train", "step": completed, "window_sha256": digest,
                      "attempt_seconds": elapsed, "attempted_tokens_total": attempted_tokens,
                      "successful_tokens_total": successful_tokens,
                      "unscaled_preclip_gradient_groups": gradient_groups if milestone else None,
                      **result})
                if completed in milestones:
                    evaluate_now()
    audit = finite_state_audit(model, optimizer, scaler)
    checkpoint_info = {"retained": False, "policy": "MQR canonical+scale both seeds; dense seed0; final only"}
    if stage == "learn" and retain_checkpoint(arm, tc.seed):
        model.cpu()
        _cpu_optimizer(optimizer)
        path = output / "last.pt"
        save_checkpoint(path, model=model, optimizer=optimizer, config=config, completed_steps=completed,
                        predicted_tokens=successful_tokens, generator=generator,
                        train_fingerprint=train.fingerprint,
                        validation_fingerprint=TokenStream(validation_tokens, config.model.context_length,
                                                           config.model.vocab_size).fingerprint,
                        scaler=scaler, data_identity=identity)
        checkpoint_info = {**checkpoint_info, "retained": True, "path": str(path),
                           **audit_checkpoint(path, config, completed_steps=completed,
                                              predicted_tokens=successful_tokens, data_identity=identity),
                           "initialization_sha256": file_sha256(output / "initialization.json"),
                           "validation_bank_sha256": validation_bank["bank_sha256"]}
        _write_json(output / "checkpoint_audit.json", checkpoint_info)
    result = {"arm": arm, "seed": tc.seed, "stage": stage, "completed_steps": completed,
              "attempted_tokens": attempted_tokens, "successful_tokens": successful_tokens,
              "successful_window_sha256": successful_window_digests,
              "amp_overflows": overflows, "initialization": initialization,
              "evaluations": evaluations, "final_state_audit": audit, "checkpoint": checkpoint_info,
              "throughput_comparison_claim": False}
    _write_json(output / "summary.json", result)
    del optimizer, scaler, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_pilot(*, data_dir: str | Path, output_root: str | Path, device: str = "cuda:0",
              seed: int = 0, microbatch: int = 1, stage: str = "learn", dry_run: bool = False,
              cpu_test: bool = False, stop_after: int = 128, arms: tuple[str, ...] | None = None) -> dict:
    """Run one seed of the registered two-seed experiment in a fresh directory."""
    if stage not in ("learn", "init"):
        raise ValueError("Pilot stage must be learn or init.")
    if type(stop_after) is not int or (not cpu_test and stop_after != 128) or not 1 <= stop_after <= 128:
        raise ValueError("The registered pilot stops at 128 updates; only CPU tests may shorten it.")
    if arms is not None and not cpu_test:
        raise ValueError("Registered GPU learning runs require all seven arms.")
    chosen_arms = ARMS if arms is None else tuple(arms)
    if not chosen_arms or len(set(chosen_arms)) != len(chosen_arms) or any(arm not in ARMS for arm in chosen_arms):
        raise ValueError("Pilot arms must be unique known arm names.")
    if chosen_arms[0] != "dense":
        raise ValueError("Dense must run first to define the train-only scale target.")
    root, output = Path(data_dir).resolve(), Path(output_root).resolve()
    if output == root or output.is_relative_to(root):
        raise ValueError("Pilot outputs must be outside the prepared data bundle.")
    identity = prepared_identity(root)
    vocab_size = load_tokenizer(root / "tokenizer.json").vocab_size
    configs = pilot_configs(vocab_size=vocab_size, seed=seed, microbatch=microbatch,
                            device=device, cpu_test=cpu_test)
    config = configs["mqr"]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    train = TokenStream.from_file(root / "train.npy", config.model.context_length, vocab_size)
    validation = TokenStream.from_file(root / "validation.npy", config.model.context_length, vocab_size)
    if train.fingerprint == validation.fingerprint:
        raise ValueError("Pilot training and validation token streams are identical.")
    train_bank = build_window_bank(train.tokens, manifest, split="train", context_length=config.model.context_length,
                                   windows_per_language=1, seed=2026102201)
    validation_bank = build_window_bank(validation.tokens, manifest, split="validation", context_length=config.model.context_length,
                                        windows_per_language=1 if cpu_test else 16, seed=2026102202)
    plan = {"protocol_version": PROTOCOL_VERSION, "stage": stage, "seed": seed,
            "required_study_seeds": [0, 1], "arms": list(chosen_arms), "fresh_weights_only": True,
            "absolute_stop_after": stop_after if stage == "learn" else 0,
            "schedule": {"max_steps": 256, "warmup_steps": 16, "learning_rate": 3e-4,
                         "min_lr_ratio": 0.1, "origin": "new pilot; never resume original calibration"},
            "effective_batch": 32, "microbatch": microbatch, "data_identity": identity,
            "train_fingerprint": train.fingerprint, "validation_fingerprint": validation.fingerprint,
            "train_probe_bank": train_bank, "validation_bank": validation_bank,
            "resolved_configs": {arm: configs[arm].to_dict() for arm in chosen_arms},
            "evaluation_steps": sorted((set(EVALUATION_STEPS) | {stop_after}) & set(range(stop_after + 1))),
            "checkpoint_retention": [arm for arm in chosen_arms if retain_checkpoint(arm, seed)] if stage == "learn" else [],
            "selection_policy": "development validation only; no model evaluation on test, translation or NLI",
            "primary_endpoint": {"step": 128, "metric": "equal-weight macro validation NLL over four languages",
                                 "comparison": "mqr_scale versus original mqr, paired within each seed"},
            "diagnostic_support_rule": {"required_seeds": [0, 1], "minimum_nll_improvement_each_seed": 0.05,
                                        "minimum_positive_gap_to_dense_closed_each_seed": 0.25,
                                        "gap_definition": "(original MQR NLL - matched MQR NLL) / (original MQR NLL - dense NLL), only if original gap is positive",
                                        "canonical_catch_up": "If canonical MQR catches dense, retain original initialization.",
                                        "scope": "diagnostic support only; no automatic default promotion or statistical significance claim"},
            "source_limits": "Small readiness corpus; random train windows repeat; domain/orthography/source limits remain. Not final quality evidence.",
            "timing_policy": "No throughput comparison claims; independent seed processes may run concurrently.",
            "cpu_correctness_test_only": cpu_test, "dry_run": dry_run}
    if dry_run:
        return plan
    if device == "cpu" and not cpu_test:
        raise ValueError("CPU execution requires --cpu-test; use --dry-run for full-shape planning.")
    hardware = inspect_device(device, cpu_test=cpu_test)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a fresh output directory; pilot runs never overwrite or resume.")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "plan.json", {**plan, "hardware": hardware})
    _write_json(output / "train_probe_bank.json", train_bank)
    _write_json(output / "validation_bank.json", validation_bank)
    (output / "data_manifest.json").write_bytes((root / "manifest.json").read_bytes())
    (output / "tokenizer.json").write_bytes((root / "tokenizer.json").read_bytes())
    source_dir = Path(__file__).parent
    _write_json(output / "source_provenance.json", {
        "python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
        "source_sha256": {name: file_sha256(source_dir / name) for name in
                          ("pilot.py", "tuning.py", "head_diagnostics.py", "training.py", "heads.py", "model.py", "config.py")}})
    results, dense_target, backbone_sha = {}, None, None
    for arm in chosen_arms:
        print(f"pilot seed={seed} arm={arm} device={device} stage={stage}", flush=True)
        result = _run_arm(configs[arm], arm, train, validation.tokens, train_bank, validation_bank,
                          output / arm, stop_after=stop_after, identity=identity,
                          dense_target_std=dense_target, stage=stage)
        initial = result["initialization"]
        if arm == "dense":
            dense_target = initial["before"]["aggregate_logits"]["centered_std"]
            backbone_sha = initial["initial_backbone_sha256"]
        if initial["initial_backbone_sha256"] != backbone_sha:
            raise ValueError("Pilot arms did not start with identical backbone weights.")
        if results and result["successful_window_sha256"] != results["dense"]["successful_window_sha256"]:
            raise ValueError("Pilot arms did not receive identical successful training windows.")
        results[arm] = result
        _write_json(output / "progress.json", {"completed_arms": list(results), "seed": seed})
    report = {**plan, "hardware": hardware, "results": results, "status": "complete",
              "model_quality_claim": False, "throughput_comparison_claim": False}
    _write_json(output / "report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, choices=(0, 1), default=0)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--stage", choices=("learn", "init"), default="learn")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cpu-test", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_pilot(**vars(args))
    except (ValueError, OSError, ImportError, FloatingPointError) as exc:
        parser.exit(2, f"Learning pilot: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
