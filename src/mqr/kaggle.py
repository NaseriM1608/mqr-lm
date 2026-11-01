"""Bounded, single-device Kaggle smoke tests and paired GPU calibration.

Synthetic runs measure implementation cost only. This helper never launches a
full research schedule and never downloads data or installs PyTorch.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig, HeadConfig, ModelConfig, TrainConfig
from .data import TokenStream, load_tokenizer
from .provenance import prepared_identity
from .tracking import read_authoritative_metrics
from .training import load_model, run_training


def session_configs(stage: str, device: str, *, config_path: str | Path | None = None,
                    vocab_size: int | None = None) -> dict[str, ExperimentConfig]:
    """Resolve matched heads without shortening their original LR schedule."""
    if stage not in {"smoke", "calibrate", "prepared"}:
        raise ValueError("stage must be smoke, calibrate, or prepared")
    if stage == "smoke":
        if config_path is not None:
            raise ValueError("Smoke uses its tiny built-in configuration; use calibrate for custom shapes.")
        base = ExperimentConfig(
            ModelConfig(vocab_size=259, layers=2, d_model=64, attention_heads=4,
                        ffn_dim=128, context_length=64),
            HeadConfig(d_e=32, d_c=64, num_queries=4),
            TrainConfig(max_steps=50, warmup_steps=5, batch_size=2,
                        checkpoint_every=0, eval_every=0, eval_batches=1, num_threads=2),
        )
    elif config_path is not None:
        base = ExperimentConfig.from_json(config_path)
    else:
        # Same full shape and original 3B-token schedule as multilingual_mqr.
        # The caller's absolute stop_after bounds this diagnostic to a few steps.
        base = ExperimentConfig(
            ModelConfig(vocab_size=65536, layers=12, d_model=512, attention_heads=8,
                        ffn_dim=2048, context_length=512),
            HeadConfig(d_e=128, d_c=256, num_queries=4),
            TrainConfig(batch_size=1, gradient_accumulation=32, max_steps=183106,
                        warmup_steps=3662, checkpoint_every=250, eval_every=250,
                        eval_batches=16, num_threads=2),
        )
    model = replace(base.model, vocab_size=vocab_size) if vocab_size is not None else base.model
    training = replace(base.training, device=device, precision="fp32" if device == "cpu" else "fp16",
                       num_threads=2)
    # Retain the seed/backbone, data sampling, optimizer and complete schedule.
    return {
        "dense": replace(base, model=model, training=training,
                         head=replace(base.head, kind="dense", num_queries=1, d_e=model.d_model)),
        "mqr": replace(base, model=model, training=training,
                       head=replace(base.head, kind="mqr")),
    }


def inspect_device(device: str, *, cpu_test: bool = False) -> dict:
    """Require a numbered CUDA device so two visible T4s are never ambiguous."""
    torch_version = tuple(int(part) for part in str(torch.__version__).split(".")[:2])
    numpy_version = tuple(int(part) for part in np.__version__.split(".")[:2])
    if sys.version_info < (3, 11) or not (2, 5) <= torch_version < (3, 0) or not (1, 26) <= numpy_version < (3, 0):
        raise ValueError("Use Python 3.11+, PyTorch >=2.5,<3 and NumPy >=1.26,<3; preserve the preinstalled CUDA torch.")
    common = {"python": platform.python_version(), "torch": str(torch.__version__),
              "torch_cuda_build": torch.version.cuda, "device": device,
              "parallelism": "one device per process; GPU memory is not pooled"}
    if device == "cpu" and cpu_test:
        return {**common, "cpu_correctness_test_only": True}
    if not device.startswith("cuda:") or not device.removeprefix("cuda:").isdigit():
        raise ValueError("Choose an explicit CUDA device such as cuda:0; CPU smoke tests require --cpu-test.")
    index = int(device.split(":")[1])
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        raise ValueError("Selected CUDA GPU is unavailable. Enable Kaggle GPU and keep its CUDA-enabled PyTorch.")
    torch.cuda.set_device(index)
    properties = torch.cuda.get_device_properties(index)
    return {**common, "visible_gpus": torch.cuda.device_count(), "name": properties.name,
            "vram_bytes": properties.total_memory, "compute_capability": list(torch.cuda.get_device_capability(index)),
            "precision": "fp16 with GradScaler", "native_bf16_requested": False}


def calibration_statistics(path: str | Path, warmup_steps: int) -> dict:
    events = read_authoritative_metrics(path)
    measured = [event for event in events if event["event"] == "train" and event["step"] > warmup_steps]
    if not measured:
        raise ValueError("No measured steps remain after warmup.")
    seconds = sum(event["step_seconds"] for event in measured)
    # The trainer reports tokens/second for the full accumulated optimizer step.
    tokens = sum(event["step_tokens_s"] * event["step_seconds"] for event in measured)
    peaks = [event["peak_vram_bytes"] for event in measured if event["peak_vram_bytes"] is not None]
    overflows = [event for event in events if event["event"] == "amp_overflow"]
    # Overflow events store the number of completed steps, so step==warmup
    # describes a failed attempt at the first measured optimizer update.
    steady_overflows = sum(event["step"] >= warmup_steps for event in overflows)
    return {"warmup_steps_excluded": warmup_steps, "measured_optimizer_steps": len(measured),
            "weighted_tokens_per_second": tokens / seconds,
            "median_step_seconds": statistics.median(event["step_seconds"] for event in measured),
            "p95_step_seconds": float(np.percentile([event["step_seconds"] for event in measured], 95)),
            "peak_allocated_vram_bytes": max(peaks) if peaks else None,
            "amp_overflow_events": len(overflows),
            "steady_amp_overflow_events": steady_overflows,
            "timing_usable_for_comparison": steady_overflows == 0,
            "warning": ("AMP-skipped attempts in the measured region have no recorded duration; "
                        "reported throughput omits their cost and cannot justify a head-speed comparison.")
                       if steady_overflows else None,
            "quality_claim": False,
            "timing_scope": ("synchronized successful optimizer steps; excludes validation, checkpoint I/O, "
                             "setup and unmeasured AMP-skipped attempts; steady-state overflows invalidate throughput comparisons")}


def _check_checkpoint(path: Path, config: ExperimentConfig) -> dict:
    # Loading on CPU verifies that downloaded checkpoints remain usable without a GPU.
    model, saved_config, state = load_model(path, "cpu")
    if saved_config != config or state["completed_steps"] != 2:
        raise ValueError("Smoke checkpoint configuration or step does not match.")
    if not all(torch.isfinite(parameter).all().item() for parameter in model.parameters()):
        raise ValueError("Checkpoint contains nonfinite model weights.")
    for key in ("optimizer", "sampler_rng", "rng", "schedule", "data_identity"):
        if key not in state:
            raise ValueError(f"Checkpoint is missing resume state: {key}")
    if config.training.precision == "fp16" and not state.get("scaler"):
        raise ValueError("FP16 checkpoint is missing GradScaler state.")
    return {"loaded_on_cpu": True, "finite_weights": True,
            "optimizer_rng_sampler_schedule_verified": True,
            "fp16_scaler_verified": config.training.precision == "fp16"}


def run_session(*, stage: str = "smoke", device: str = "cuda:0",
                output_root: str | Path = "/kaggle/working/mqr-smoke",
                config_path: str | Path | None = None, data_dir: str | Path | None = None,
                steps: int | None = None, warmup_steps: int | None = None,
                dry_run: bool = False, cpu_test: bool = False) -> dict:
    """Run two bounded heads, including a checkpoint/restart check for each."""
    steps = (4 if stage == "smoke" else 120) if steps is None else steps
    warmup_steps = (2 if stage == "smoke" else 20) if warmup_steps is None else warmup_steps
    if type(steps) is not int or not 4 <= steps <= 1000:
        raise ValueError("Diagnostic steps must be between 4 and 1000; this helper cannot launch a full study.")
    if type(warmup_steps) is not int or not 0 <= warmup_steps < steps:
        raise ValueError("warmup_steps must leave at least one measured step.")
    if stage == "prepared" and data_dir is None:
        raise ValueError("Prepared calibration requires --data-dir.")
    if stage != "prepared" and data_dir is not None:
        raise ValueError("Use stage=prepared to consume a real data bundle.")
    if cpu_test and (device != "cpu" or stage != "smoke"):
        raise ValueError("--cpu-test is only for the tiny CPU smoke stage.")
    identity = None
    vocab = None
    if data_dir is not None:
        identity = prepared_identity(data_dir)
        vocab = load_tokenizer(Path(data_dir) / "tokenizer.json").vocab_size
    configs = session_configs(stage, device, config_path=config_path, vocab_size=vocab)
    if steps > configs["mqr"].training.max_steps:
        raise ValueError("Diagnostic steps exceed the original training schedule.")
    output = Path(output_root).resolve()
    if data_dir is not None:
        bundle_root = Path(data_dir).resolve()
        if output == bundle_root or output.is_relative_to(bundle_root):
            raise ValueError("Outputs must be outside the attached read-only data bundle.")
    report = {"stage": stage, "purpose": "GPU correctness and cost calibration; not language-model quality evidence",
              "absolute_stop_after": steps, "warmup_steps": warmup_steps,
              "output_root": str(output), "data_identity": identity,
              "actual_tokenizer_vocab_size": vocab,
              "vocabulary_policy": "verified tokenizer size replaces the template ceiling for both heads" if vocab is not None else "configured synthetic vocabulary",
              "resolved_configs": {key: value.to_dict() for key, value in configs.items()},
              "paired_execution": "sequential on the explicitly selected GPU",
              "dry_run": dry_run}
    if dry_run:
        return {**report, "hardware_execution": "skipped; GPU compatibility is not validated by this dry run"}
    if device == "cpu" and not cpu_test:
        raise ValueError("CPU execution is restricted to --cpu-test smoke; use --dry-run for full-shape plans.")
    report["hardware"] = inspect_device(device, cpu_test=cpu_test)
    if any((output / kind).exists() for kind in configs) or (output / "report.json").exists():
        raise ValueError("Choose a fresh output directory; existing calibration runs are never overwritten.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "plan.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if data_dir is not None:
        # Preserve source-selection metadata and tokenizer alongside downloaded logs.
        for source_name, output_name in (("manifest.json", "data_manifest.json"),
                                         ("tokenizer.json", "tokenizer.json")):
            (output / output_name).write_bytes((Path(data_dir) / source_name).read_bytes())
    config = configs["mqr"]
    length, vocab = config.model.context_length, config.model.vocab_size
    if data_dir is not None:
        train = TokenStream.from_file(Path(data_dir) / "train.npy", length, vocab)
        validation = TokenStream.from_file(Path(data_dir) / "validation.npy", length, vocab)
    else:
        rng = np.random.default_rng(20261022)
        count = max(8192, 16 * (length + 1))
        train = TokenStream(rng.integers(vocab, size=count, dtype=np.uint32), length, vocab)
        validation = TokenStream(rng.integers(vocab, size=count, dtype=np.uint32), length, vocab)
        identity = {"kind": "synthetic_calibration", "seed": 20261022,
                    "token_distribution": "uniform random; no language-model quality claims"}
    results = {}
    for kind, config in configs.items():
        run_dir = output / kind
        print(f"{kind}: {device}, {config.training.precision}; first 2 steps then resume to {steps}", flush=True)
        run_training(config, train, validation, run_dir, stop_after=2, data_identity=identity)
        checkpoint_checks = _check_checkpoint(run_dir / "last.pt", config)
        summary = run_training(config, train, validation, run_dir, resume=run_dir / "last.pt",
                               stop_after=steps, data_identity=identity)
        if summary["completed_steps"] != steps:
            raise ValueError("Resume did not reach the requested bounded step.")
        results[kind] = {"resume_check": {**checkpoint_checks, "continued_from_step_2_to": steps},
                         "training": summary,
                         "calibration": calibration_statistics(run_dir / "metrics.jsonl", warmup_steps)}
        # A subsequent head should not inherit allocator reservations in reports.
        if device.startswith("cuda:"):
            torch.cuda.empty_cache()
    report["results"] = results
    report["data_identity"] = identity
    report["dry_run"] = False
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "calibrate", "prepared"), default="smoke")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default="/kaggle/working/mqr-smoke")
    parser.add_argument("--config", dest="config_path")
    parser.add_argument("--data-dir")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cpu-test", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run_session(**vars(args))
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"Kaggle calibration: {exc}\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
