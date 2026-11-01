"""Preregistered numerical gates for chunk scheduling; never a speed benchmark.

The inner readiness archive is the source trust anchor. No installed/local mqr
package supplies the implementation. A CPU dry run is never GPU evidence.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import gc
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import platform
import sys
import tempfile
import time
from types import SimpleNamespace
import uuid
import zipfile

import numpy as np
import torch
import torch.nn.functional as F


ARCHIVE_SHA256 = "2dacdd37bc4044889e3c3cd613951c3be3c98bd1679bcc7735485eb378dbd649"
PACKAGE_MANIFEST_SHA256 = "74b4ddb2039bf242f2fa3f82318e1f265ed2a06100d89ab52b9be13be9bc9e0d"
CHUNKS = (64, 128, 256, 512)
ARMS = ("dense", "mqr", "mqr_scale")
MASK_POSITIONS = (0, 63, 64, 127, 128, 255, 256, 511, 512, 1023)
PROBE_POSITIONS = 1025
PROBE_SEED = 77431
MODEL_SEED = 0
HEAD_GRADIENT_SCALE = 1024.0
STRESS_FEATURE_SCALE = 32.0
TOLERANCES = {
    "fp16_loss_abs": 2e-4,
    "gradient_relative_l2": 0.01,
    "gradient_cosine_min": 0.9999,
    "fp64_oracle_loss_abs": 2e-12,
    "fp64_oracle_gradient_relative_l2": 2e-11,
    "frozen_fp32_ce_loss_abs": 2e-6,
    "frozen_fp32_ce_gradient_relative_l2": 2e-6,
    "trajectory_parameter_relative_l2": 2e-5,
    "trajectory_parameter_max_abs": 2e-6,
    "trajectory_moment_relative_l2": 0.02,
    "trajectory_moment_cosine_min": 0.9995,
}
PROTOCOL = {
    "version": "mqr-chunk-numerics-v1",
    "chunks": list(CHUNKS), "arms": list(ARMS), "model_seed": MODEL_SEED,
    "feature_positions": PROBE_POSITIONS, "probe_seed": PROBE_SEED,
    "mask_positions": list(MASK_POSITIONS), "stress_feature_scale": STRESS_FEATURE_SCALE,
    "head_gradient_scale": HEAD_GRADIENT_SCALE,
    "scale_calibration": "64 training-only FP32 feature positions:16 from each en/de/ar/fa window; multiply final query projection weight+bias only",
    "head_checkpoint_chunks": True,
    "head_reference": "same frozen features/weights; FP32 full-vocabulary CE, chunk64",
    "fp32_error_policy": "candidate error <= baseline FP16 error + declared FP16 allowance, per group",
    "zero_policy": "zero reference groups require exactly zero candidate groups",
    "trajectory_updates": 3, "trajectory_global_batch": 32,
    "trajectory_microbatch": 8, "trajectory_accumulation": 4,
    "trajectory_arms": list(ARMS),
    "trajectory_masks": "microbatch0:first target; microbatch1:first row; microbatch2:first2rows; microbatch3:first4rows plus final target",
    "trajectory_valid_targets_per_microbatch": [4095, 3584, 3072, 2047],
    "trajectory_lr_schedule": "unchanged original 183106 updates / 3662 warmup",
    "tolerances": TOLERANCES,
    "cpu_note": "FP64 oracle is separate; frozen exact_cross_entropy intentionally casts CE logits to FP32",
    "quality_claim": False, "speed_claim": False,
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _safe_member(name, info):
    path = PurePosixPath(name)
    _require(not path.is_absolute() and ".." not in path.parts and "\\" not in name
             and ":" not in name and "\x00" not in name and not info.is_dir()
             and (info.external_attr >> 16) & 0o170000 != 0o120000, "Unsafe ZIP member")


@contextmanager
def verified_frozen_modules(archive_path):
    """Verify every payload before importing individually written source files."""
    archive_path = Path(archive_path).resolve()
    _require(_sha(archive_path) == ARCHIVE_SHA256, "Frozen inner archive SHA-256 differs")
    namespace = "_mqr_chunk_numerics_" + uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix="mqr-chunk-source-") as temporary:
        source_root = Path(temporary)
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            _require(len(names) == len(set(names)), "Duplicate ZIP names")
            manifest_raw = archive.read("package_manifest.json")
            _require(hashlib.sha256(manifest_raw).hexdigest() == PACKAGE_MANIFEST_SHA256,
                     "Frozen package manifest SHA-256 differs")
            manifest = json.loads(manifest_raw)
            _require(set(names) == set(manifest["files"]) | {"package_manifest.json"},
                     "Archive members differ from package manifest")
            sources = {}
            for name, expected in manifest["files"].items():
                info = archive.getinfo(name)
                _safe_member(name, info)
                raw = archive.read(info)
                _require(len(raw) == expected["bytes"] and hashlib.sha256(raw).hexdigest() == expected["sha256"],
                         f"Frozen payload differs: {name}")
                if name.startswith("src/mqr/") and name.endswith(".py"):
                    target = source_root.joinpath(*PurePosixPath(name).parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(raw)
                    sources[name.removeprefix("src/mqr/")] = expected["sha256"]
        root = source_root / "src/mqr"
        spec = importlib.util.spec_from_file_location(namespace, root / "__init__.py",
                                                     submodule_search_locations=[str(root)])
        _require(spec is not None and spec.loader is not None, "Cannot load frozen module")
        package = importlib.util.module_from_spec(spec)
        sys.modules[namespace] = package
        try:
            spec.loader.exec_module(package)
            modules = {name: importlib.import_module(namespace + "." + name) for name in
                       ("heads", "training", "config", "model", "tuning", "pilot", "kaggle",
                        "head_diagnostics", "provenance", "data")}
            binding = {"archive_sha256": ARCHIVE_SHA256,
                       "package_manifest_sha256": PACKAGE_MANIFEST_SHA256,
                       "source_sha256": sources, "expected_data_identity": manifest["prepared_identity"]}
            yield SimpleNamespace(**modules), binding
        finally:
            for name in list(sys.modules):
                if name == namespace or name.startswith(namespace + "."):
                    del sys.modules[name]


def _snapshot(module, features=None):
    values = {name: (parameter.grad.detach().cpu().clone()
                     if parameter.grad is not None else torch.zeros_like(parameter, device="cpu", dtype=torch.float32))
              for name, parameter in module.named_parameters()}
    if features is not None:
        values["features"] = (features.grad.detach().cpu().clone() if features.grad is not None
                              else torch.zeros_like(features, device="cpu"))
    return values


def _group(name):
    if name == "features":
        return "features"
    if name.startswith("backbone.embedding."):
        return "backbone.embedding"
    if name.startswith("backbone."):
        if ".attention." in name:
            return "backbone.attention"
        if ".ffn." in name:
            return "backbone.ffn"
        return "backbone.norm"
    local = name.removeprefix("head.")
    return "head." + local.split(".")[0]


def group_errors(reference, candidate):
    """Streaming CPU FP64 statistics, without concatenation or serialized arrays."""
    _require(set(reference) == set(candidate), "Gradient/state tensor names differ")
    accumulators = {}
    for name in reference:
        left, right = reference[name], candidate[name]
        _require(left.shape == right.shape, f"Tensor shape differs: {name}")
        row = accumulators.setdefault(_group(name), {"reference_square": 0.0, "candidate_square": 0.0,
             "difference_square": 0.0, "dot": 0.0, "max_abs_difference": 0.0,
             "elements": 0, "finite": True})
        a, b = left.reshape(-1), right.reshape(-1)
        for start in range(0, a.numel(), 262144):
            x, y = a[start:start + 262144].double(), b[start:start + 262144].double()
            finite = bool(torch.isfinite(x).all() and torch.isfinite(y).all())
            row["finite"] = row["finite"] and finite
            row["elements"] += x.numel()
            if not finite:
                continue
            difference = x - y
            row["reference_square"] += float(x.square().sum())
            row["candidate_square"] += float(y.square().sum())
            row["difference_square"] += float(difference.square().sum())
            row["dot"] += float((x * y).sum())
            row["max_abs_difference"] = max(row["max_abs_difference"], float(difference.abs().max()))
    result = {}
    for group, row in accumulators.items():
        a, b, d = (math.sqrt(row[key]) for key in ("reference_square", "candidate_square", "difference_square"))
        zero = a == 0
        result[group] = {"elements": row["elements"], "finite": row["finite"],
            "reference_l2": a, "candidate_l2": b, "difference_l2": d,
            "reference_rms": a / math.sqrt(row["elements"]),
            "max_abs_difference": row["max_abs_difference"],
            "relative_l2": d / a if a else (0.0 if b == 0 else None),
            "cosine": max(-1.0, min(1.0, row["dot"] / (a * b))) if a and b else (1.0 if a == b == 0 else 0.0),
            "zero_reference": zero, "zero_preserved": not zero or b == 0}
    return result


def gradient_gate(groups, *, relative=None, cosine=None, max_abs=None):
    relative = TOLERANCES["gradient_relative_l2"] if relative is None else relative
    cosine = TOLERANCES["gradient_cosine_min"] if cosine is None else cosine
    return bool(groups) and all(row["finite"] and row["zero_preserved"]
        and row["relative_l2"] is not None and row["relative_l2"] <= relative
        and row["cosine"] >= cosine
        and (max_abs is None or row["max_abs_difference"] <= max_abs) for row in groups.values())


def _head_pass(head, features, targets, loss_function, *, precision="fp32", gradient_scale=1.0):
    head.zero_grad(set_to_none=True)
    f = features.detach().clone().requires_grad_(True)
    context = torch.autocast("cuda", dtype=torch.float16) if precision == "fp16" else torch.autocast("cpu", enabled=False)
    with context:
        loss = loss_function(head, f, targets)
    if not torch.isfinite(loss).item():
        raise FloatingPointError("Nonfinite numerical-probe loss")
    (loss * gradient_scale).backward()
    with torch.no_grad():
        for parameter in head.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(gradient_scale)
        if f.grad is not None:
            f.grad.div_(gradient_scale)
    gradients = _snapshot(head, f)
    head.zero_grad(set_to_none=True)
    return {"loss": float(loss.detach()), "gradients": gradients}


def _fp64_ce(head, features, targets, chunk=None):
    features, targets = features.reshape(-1, features.shape[-1]), targets.reshape(-1)
    count = int((targets != -100).sum())
    if not count:
        return features.sum() * 0
    chunk = len(targets) if chunk is None else chunk
    losses = [F.cross_entropy(head.score_all(features[i:i + chunk]), targets[i:i + chunk],
                              reduction="sum", ignore_index=-100) for i in range(0, len(targets), chunk)]
    return torch.stack(losses).sum() / count


def cpu_gate(modules):
    """True FP64 oracle plus the untouched frozen loss's intentional FP32 CE."""
    torch.manual_seed(9187)
    features = torch.randn(PROBE_POSITIONS, 12, dtype=torch.float64)
    targets = torch.randint(0, 37, (PROBE_POSITIONS,))
    targets[list(MASK_POSITIONS)] = -100
    cases = []
    for kind in ("dense", "mqr"):
        config = modules.config.HeadConfig(kind=kind, d_e=5, d_c=9, num_queries=4)
        head = modules.heads.build_head(config, 12, 37).double()
        for all_masked in (False, True):
            y = torch.full_like(targets, -100) if all_masked else targets
            reference = _head_pass(head, features, y, _fp64_ce)
            for chunk in CHUNKS:
                oracle = _head_pass(head, features, y, lambda h, f, t: _fp64_ce(h, f, t, chunk))
                # Frozen code casts CE logits to FP32 even for FP64 heads/features.
                frozen = _head_pass(head, features, y, lambda h, f, t: modules.training.exact_cross_entropy(h, f, t, chunk))
                oracle_errors = group_errors(reference["gradients"], oracle["gradients"])
                frozen_errors = group_errors(reference["gradients"], frozen["gradients"])
                oracle_delta, frozen_delta = abs(reference["loss"] - oracle["loss"]), abs(reference["loss"] - frozen["loss"])
                passed = (oracle_delta <= TOLERANCES["fp64_oracle_loss_abs"]
                    and gradient_gate(oracle_errors, relative=TOLERANCES["fp64_oracle_gradient_relative_l2"], cosine=1 - 2e-12)
                    and frozen_delta <= TOLERANCES["frozen_fp32_ce_loss_abs"]
                    and gradient_gate(frozen_errors, relative=TOLERANCES["frozen_fp32_ce_gradient_relative_l2"], cosine=1 - 2e-10))
                cases.append({"head": kind, "chunk": chunk, "all_masked": all_masked,
                    "oracle_loss": reference["loss"], "oracle_loss_delta": oracle_delta,
                    "frozen_fp32_ce_loss_delta": frozen_delta, "oracle_groups": oracle_errors,
                    "frozen_fp32_ce_groups": frozen_errors, "status": "passed" if passed else "failed"})
    return {"status": "passed" if all(row["status"] == "passed" for row in cases) else "failed",
            "device": "cpu", "head_features_dtype": "float64", "positions": PROBE_POSITIONS,
            "mask_positions": list(MASK_POSITIONS), "cases": cases,
            "fp64_loss_oracle_separate_from_frozen_fp32_ce": True}


def _fp32_regression(candidate, baseline):
    """Candidate reference error may grow by only the preregistered allowance."""
    if candidate["loss_delta"] > baseline["loss_delta"] + TOLERANCES["fp16_loss_abs"]:
        return False
    if set(candidate["groups"]) != set(baseline["groups"]):
        return False
    for name, row in candidate["groups"].items():
        base = baseline["groups"][name]
        if (not row["finite"] or not row["zero_preserved"] or row["relative_l2"] is None
                or base["relative_l2"] is None
                or row["relative_l2"] > base["relative_l2"] + TOLERANCES["gradient_relative_l2"]
                or row["cosine"] < base["cosine"] - (1 - TOLERANCES["gradient_cosine_min"])):
            return False
    return True


def head_chunk_gate(modules, head, features, targets, *, check_deadline=lambda: None):
    check_deadline()
    reference = _head_pass(head, features, targets,
        lambda h, f, y: modules.training.exact_cross_entropy(h, f, y, 64))
    baseline = _head_pass(head, features, targets,
        lambda h, f, y: modules.training.exact_cross_entropy(h, f, y, 64),
        precision="fp16", gradient_scale=HEAD_GRADIENT_SCALE)
    base_reference_error = {"loss_delta": abs(reference["loss"] - baseline["loss"]),
                            "groups": group_errors(reference["gradients"], baseline["gradients"])}
    rows = []
    for chunk in CHUNKS:
        check_deadline()
        candidate = baseline if chunk == 64 else _head_pass(head, features, targets,
            lambda h, f, y: modules.training.exact_cross_entropy(h, f, y, chunk),
            precision="fp16", gradient_scale=HEAD_GRADIENT_SCALE)
        errors = group_errors(baseline["gradients"], candidate["gradients"])
        ref_error = {"loss_delta": abs(reference["loss"] - candidate["loss"]),
                     "groups": group_errors(reference["gradients"], candidate["gradients"])}
        loss_delta = abs(candidate["loss"] - baseline["loss"])
        finite = all(math.isfinite(candidate["loss"]) and row["finite"] for row in errors.values())
        relative_pass = gradient_gate(errors)
        fp32_pass = _fp32_regression(ref_error, base_reference_error)
        passed = finite and loss_delta <= TOLERANCES["fp16_loss_abs"] and relative_pass and fp32_pass
        rows.append({"chunk": chunk, "status": "passed" if passed else "failed", "finite": finite,
            "loss": candidate["loss"], "loss_delta_vs_fp16_chunk64": loss_delta,
            "gradient_groups_vs_fp16_chunk64": errors,
            "gradient_gate_passed": relative_pass, "fp32_reference_error": ref_error,
            "fp32_error_regression_gate_passed": fp32_pass})
        if chunk != 64:
            del candidate
    return {"status": "passed" if all(row["status"] == "passed" for row in rows) else "failed",
            "fp32_reference_loss": reference["loss"], "fp16_baseline_reference_error": base_reference_error,
            "head_gradient_scale": HEAD_GRADIENT_SCALE, "gates": rows}


def _state_snapshot(model, optimizer):
    parameters = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
    moments = {key: {} for key in ("exp_avg", "exp_avg_sq")}
    steps = set()
    for name, parameter in model.named_parameters():
        state = optimizer.state[parameter]
        steps.add(int(state["step"]))
        for key in moments:
            moments[key][name] = state[key].detach().cpu().clone()
    return {"parameters": parameters, **moments, "steps": sorted(steps)}


def trajectory_gate(modules, config, batches, *, gain=1.0, check_deadline=lambda: None):
    """Three actual fixed-global-batch updates; no retry/window substitution."""
    baseline_gradients, baseline_events, baseline_state = [], [], None
    rows = []
    for chunk in CHUNKS:
        check_deadline()
        trial_config = replace(config, training=replace(config.training, loss_chunk_positions=chunk))
        model, optimizer, scaler, _, device = modules.tuning._state(trial_config)
        if gain != 1.0:
            with torch.no_grad():
                model.head.query_projection.weight.mul_(gain)
                model.head.query_projection.bias.mul_(gain)
        events, gradient_records = [], []
        for step, (x, y) in enumerate(batches):
            check_deadline()
            def before_clip(current, step=step):
                captured = _snapshot(current)
                if chunk == 64:
                    baseline_gradients.append(captured)
                else:
                    errors = group_errors(baseline_gradients[step], captured)
                    gradient_records.append({"update": step + 1, "groups": errors,
                                             "passed": gradient_gate(errors)})
            event = modules.tuning.optimizer_attempt(model, optimizer, scaler, trial_config,
                                                      x, y, step, before_clip=before_clip)
            _require(not event["skipped"], "Trajectory AMP overflow; numerical gate fails without retry")
            if chunk == 64:
                errors = group_errors(baseline_gradients[step], baseline_gradients[step])
                gradient_records.append({"update": step + 1, "groups": errors,
                                         "passed": gradient_gate(errors)})
            baseline_event = event if chunk == 64 else baseline_events[step]
            event["window_sha256"] = modules.tuning.window_digest(x, y)
            event["loss_delta_vs_chunk64"] = abs(event["full_nll"] - baseline_event["full_nll"])
            event["scaler_matches_baseline"] = event["scaler_scale"] == baseline_event["scaler_scale"]
            events.append(event)
        state = _state_snapshot(model, optimizer)
        if chunk == 64:
            baseline_state, baseline_events = state, events
        parameter_errors = group_errors(baseline_state["parameters"], state["parameters"])
        moment_errors = {key: group_errors(baseline_state[key], state[key]) for key in ("exp_avg", "exp_avg_sq")}
        state_passed = (state["steps"] == [3]
            and gradient_gate(parameter_errors, relative=TOLERANCES["trajectory_parameter_relative_l2"],
                              cosine=TOLERANCES["gradient_cosine_min"], max_abs=TOLERANCES["trajectory_parameter_max_abs"])
            and all(gradient_gate(error, relative=TOLERANCES["trajectory_moment_relative_l2"],
                                  cosine=TOLERANCES["trajectory_moment_cosine_min"]) for error in moment_errors.values()))
        passed = (state_passed and all(row["passed"] for row in gradient_records)
            and all(math.isfinite(e["full_nll"]) and e["gradient_norm"] is not None
                    and e["loss_delta_vs_chunk64"] <= TOLERANCES["fp16_loss_abs"]
                    and e["scaler_matches_baseline"] for e in events))
        rows.append({"chunk": chunk, "status": "passed" if passed else "failed", "config": trial_config.to_dict(),
            "updates": events, "preclip_gradient_gates": gradient_records,
            "final_parameter_groups": parameter_errors, "final_moment_groups": moment_errors,
            "optimizer_steps": state["steps"], "state_gate_passed": state_passed})
        del model, optimizer, scaler
        if chunk != 64:
            del state
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {"status": "passed" if all(row["status"] == "passed" for row in rows) else "failed",
            "initial_query_projection_gain": gain, "gates": rows}


def _feature_bank(modules, tokens, manifest, dense_config, device):
    bank = modules.pilot.build_window_bank(tokens, manifest, split="train", context_length=512,
                                           windows_per_language=1, seed=PROBE_SEED)
    modules.training.set_seed(MODEL_SEED)
    backbone = modules.model.DecoderBackbone(dense_config.model).to(device).eval()
    x, y = modules.pilot.bank_tensors(tokens, bank)
    selected = torch.linspace(0, 511, 257).long()
    features, targets, positions = [], [], []
    with torch.no_grad():
        for i, (bx, by) in enumerate(zip(x, y)):
            output, _ = backbone(bx[None].to(device))
            features.append(output[0, selected.to(device)].float())
            targets.append(by[selected].to(device))
            positions.extend({"window": i, "position": int(position)} for position in selected)
    del backbone
    features, targets = torch.cat(features)[:PROBE_POSITIONS], torch.cat(targets)[:PROBE_POSITIONS]
    targets = targets.clone()
    targets[list(MASK_POSITIONS)] = -100
    position_sha = _json_sha(positions[:PROBE_POSITIONS])
    metadata = {"bank": bank, "selected_position_count": PROBE_POSITIONS,
        "position_selection": "linspace(0,511,257) per en/de/ar/fa window; concatenate and take first1025",
        "selected_positions_sha256": position_sha, "mask_positions": list(MASK_POSITIONS),
        "valid_targets": int((targets != -100).sum()), "decoder_features": "shared untrained FP32 decoder, no autocast",
        "features_sha256": hashlib.sha256(features.cpu().contiguous().numpy().tobytes()).hexdigest(),
        "targets_sha256": hashlib.sha256(targets.cpu().contiguous().numpy().tobytes()).hexdigest()}
    return features, targets, metadata


def _aggregate_std(head, features):
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(features), 32):
            logits = head.score_all(features[start:start + 32]).float()
            total += float((logits - logits.mean(-1, keepdim=True)).square().sum())
            count += logits.numel()
    return math.sqrt(total / count)


def _component_statistics(head, features):
    maximum, finite, elements = 0.0, True, 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for start in range(0, len(features), 64):
            scores = head.component_scores(features[start:start + 64])
            finite = finite and bool(torch.isfinite(scores).all())
            if finite:
                maximum = max(maximum, float(scores.abs().max()))
            elements += scores.numel()
    return {"dtype": "float16", "finite": finite, "max_abs": maximum,
            "component_elements_examined": elements}


def _required_gates(report):
    records = [{"name": "cpu_fp64_and_frozen_fp32_ce", "status": report.get("cpu", {}).get("status", "missing")}]
    for arm in ARMS:
        for stress in (False, True):
            name = arm + ("_stress" if stress else "")
            case = report.get("head_cases", {}).get(name)
            records.append({"name": "head_" + name, "status": case["status"] if case else "missing"})
        trajectory = report.get("trajectory_cases", {}).get(arm)
        records.append({"name": "trajectory_" + arm, "status": trajectory["status"] if trajectory else "missing"})
    return records


def run_numerics(data_dir, output_dir, device="cuda:0", *, archive_path=None,
                 timeout_seconds=540, dry_run=False):
    """Write numerics.json; only a complete actual CUDA run may report passed."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    _require(0 < timeout_seconds <= 540, "Numerical worker timeout must be in (0,540] seconds")
    if archive_path is None:
        archive_path = Path(__file__).resolve().parents[1] / "tuning_v1/mqr-kaggle-readiness.zip"
    report = {"status": "running", "all_required_gates_passed": False,
              "protocol": PROTOCOL, "protocol_sha256": _json_sha(PROTOCOL),
              "helper_sha256": _sha(__file__), "tested_chunks": list(CHUNKS),
              "device": device, "dry_run": dry_run, "head_cases": {}, "trajectory_cases": {},
              "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                          "torch_cuda_build": torch.version.cuda, "scope": "CPU dry run" if dry_run else "actual CUDA FP16 numerical gate"}}
    def persist():
        report["elapsed_seconds"] = time.monotonic() - started
        report["required_gates"] = _required_gates(report)
        _write(output / "numerics.json", report)
    def check_deadline():
        if time.monotonic() - started >= timeout_seconds:
            raise TimeoutError("Preregistered numerical runtime budget exceeded")
    try:
        with verified_frozen_modules(archive_path) as (modules, binding):
            report.update({key: value for key, value in binding.items() if key != "expected_data_identity"})
            identity = modules.provenance.prepared_identity(data_dir)
            _require(identity == binding["expected_data_identity"], "Prepared data differs from frozen readiness package")
            stream = modules.data.TokenStream.from_file(Path(data_dir) / "train.npy", 512, 65536)
            manifest = json.loads((Path(data_dir) / "manifest.json").read_text(encoding="utf-8"))
            report.update(data_identity=identity, train_fingerprint=stream.fingerprint)
            torch.set_num_threads(2)
            report["cpu"] = cpu_gate(modules)
            _require(report["cpu"]["status"] == "passed", "CPU oracle/frozen CE correctness gate failed")
            bank = modules.pilot.build_window_bank(stream.tokens, manifest, split="train", context_length=512,
                                                   windows_per_language=1, seed=PROBE_SEED)
            report["training_probe_bank"] = bank
            persist()
            if dry_run:
                report["status"] = "ready_for_gpu"
                report["runtime"]["scope"] = "byte-verified frozen source/data + CPU correctness only; no CUDA gates run"
                persist()
                return report
            _require(device.startswith("cuda:") and device[5:].isdigit() and torch.cuda.is_available(),
                     "Full numerical gate requires an available explicit CUDA device")
            report["runtime"].update(modules.kaggle.inspect_device(device))
            report["runtime"].update({"float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()})
            configs = {name: modules.tuning.batch_config(config, 8) for name, config in
                       modules.kaggle.session_configs("prepared", device, vocab_size=65536).items()}
            features, targets, metadata = _feature_bank(modules, stream.tokens, manifest, configs["dense"], device)
            report["feature_probe"] = metadata
            calibration_indices = torch.cat([start + torch.linspace(0, 253, 16).long()
                                             for start in (0, 257, 514, 771)]).to(device)
            calibration_features = features[calibration_indices]
            gains = {"dense": 1.0, "mqr": 1.0}
            calibration = None
            dense_std = None
            for arm in ARMS:
                check_deadline()
                config = configs["dense" if arm == "dense" else "mqr"]
                modules.training.set_seed(MODEL_SEED)
                model = modules.model.LanguageModel(config)
                head = model.head.to(device)
                del model
                if arm == "dense":
                    dense_std = _aggregate_std(head, calibration_features)
                if arm == "mqr_scale":
                    calibration = modules.head_diagnostics.calibrate_initial_query_gain(
                        head, calibration_features, dense_std, chunk_positions=32)
                    gains[arm] = calibration["query_projection_gain"]
                    report["initialization_calibration"] = {**calibration, "positions": 64,
                        "training_only": True, "probe_selection": "16 positions per training language window",
                        "feature_indices": calibration_indices.cpu().tolist(),
                        "dense_target_centered_logit_std": dense_std}
                for stress in (False, True):
                    check_deadline()
                    scale = STRESS_FEATURE_SCALE if stress else 1.0
                    component_stats = _component_statistics(head, features * scale)
                    _require(component_stats["finite"], "Nonfinite FP16 component scores in numerical probe")
                    case = head_chunk_gate(modules, head, features * scale, targets, check_deadline=check_deadline)
                    case.update(config=config.to_dict(), feature_scale=scale,
                                component_score_statistics=component_stats,
                                initial_query_projection_gain=gains.get(arm, 1.0),
                                head_state_sha256=modules.head_diagnostics.initialization_fingerprint(head))
                    report["head_cases"][arm + ("_stress" if stress else "")] = case
                    persist()
                del head
                gc.collect()
                torch.cuda.empty_cache()
            del features, targets, calibration_features
            generator = torch.Generator(device="cpu").manual_seed(MODEL_SEED + 101)
            batches = []
            unmasked_digests = []
            for _ in range(3):
                x, y = stream.sample(32, generator)
                unmasked_digests.append(modules.tuning.window_digest(x, y))
                y[0, 0] = -100
                y[8, :] = -100
                y[16:18, :] = -100
                y[24:28, :] = -100
                y[31, -1] = -100
                batches.append((x, y))
            report["trajectory_unmasked_window_sha256"] = unmasked_digests
            report["trajectory_window_sha256"] = [modules.tuning.window_digest(x, y) for x, y in batches]
            report["trajectory_mask_policy"] = {"masked_targets_per_global_batch": 3586,
                "first_target": [0, 0], "fully_masked_rows": [8, 16, 17, 24, 25, 26, 27],
                "final_target": [31, 511], "valid_targets_per_microbatch": [4095, 3584, 3072, 2047],
                "purpose": "unequal valid counts exercise optimizer_attempt's global-token weighting"}
            for arm in ARMS:
                config = configs["dense" if arm == "dense" else "mqr"]
                report["trajectory_cases"][arm] = trajectory_gate(modules, config, batches,
                    gain=gains[arm], check_deadline=check_deadline)
                persist()
            records = _required_gates(report)
            report["all_required_gates_passed"] = (len(records) == 10 and all(row["status"] == "passed" for row in records))
            report["status"] = "passed" if report["all_required_gates_passed"] else "failed"
            check_deadline()
    except Exception as exc:
        report["status"] = "failed"
        report["all_required_gates_passed"] = False
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    persist()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--archive", "--archive-path", dest="archive_path")
    parser.add_argument("--timeout-seconds", type=float, default=540)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = run_numerics(**vars(args))
    print(json.dumps({"status": report["status"], "all_required_gates_passed": report["all_required_gates_passed"],
                      "report": str((Path(args.output_dir) / "numerics.json").resolve())}))
    return 0 if report["status"] in ("passed", "ready_for_gpu") else 2


if __name__ == "__main__":
    raise SystemExit(main())
