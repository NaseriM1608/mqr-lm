"""CPU initialization diagnostics; no optimizer updates or canonical-code changes.

All heads see one fixed decoder-feature tensor and the same shifted targets.
Experimental fan-in and calibrated-gain variants change only cloned initial
parameters. They are controls for a future validation pilot, not selected models.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .config import ExperimentConfig, HeadConfig
from .heads import DenseHead, DirectHead, MQRHead, VocabularyHead, build_head
from .model import DecoderBackbone
from .provenance import file_sha256


def gelu_gaussian_moments(variance: float, quadrature_points: int = 128) -> dict[str, float]:
    """Gaussian mean/second moment of exact GELU and its derivative by quadrature."""
    if not math.isfinite(variance) or variance < 0:
        raise ValueError("variance must be finite and nonnegative")
    if type(quadrature_points) is not int or not 16 <= quadrature_points <= 256:
        raise ValueError("quadrature_points must be an integer from 16 to 256")
    nodes, weights = np.polynomial.hermite.hermgauss(quadrature_points)
    z = math.sqrt(2.0 * variance) * nodes
    cdf = np.asarray([0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0))) for value in z])
    density = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    values, derivatives = z * cdf, cdf + z * density
    weights = weights / math.sqrt(math.pi)
    return {"mean": float(np.sum(weights * values)), "second_moment": float(np.sum(weights * values * values)),
            "derivative_second_moment": float(np.sum(weights * derivatives * derivatives))}


def variance_prediction(
    kind: str, d_model: int, d_e: int, d_c: int, num_queries: int, *,
    feature_second_moment: float = 1.0, concept_std: float = 0.02,
    query_std: float = 0.02, output_std: float = 0.02,
) -> dict:
    """Mean-field initialization predictions, with small-score LSE linearization.

    This averages random independent weights, assumes zero biases, and ignores
    finite-width/dependence corrections. Gradient ratios additionally assume
    nearly uniform branch responsibilities and a common output-row variance.
    """
    if kind not in ("dense", "direct", "mqr"):
        raise ValueError("unknown head kind")
    if any(type(size) is not int or size < 1 for size in (d_model, d_e, d_c, num_queries)):
        raise ValueError("dimensions/query count must be positive integers")
    for value in (feature_second_moment, concept_std, query_std, output_std):
        if not math.isfinite(value) or value < 0:
            raise ValueError("moments and initialization scales must be finite and nonnegative")
    moments = None
    preactivation_variance = None
    if kind == "dense":
        query_second = feature_second_moment
        component = d_model * query_second * output_std ** 2
        aggregate = component
        gradient_ratio = 1.0
    elif kind == "direct":
        query_second = d_model * feature_second_moment * query_std ** 2
        component = d_e * query_second * output_std ** 2
        aggregate = component / num_queries
        gradient_ratio = math.sqrt(d_e * query_std ** 2 / num_queries)
    else:
        preactivation_variance = d_model * feature_second_moment * concept_std ** 2
        moments = gelu_gaussian_moments(preactivation_variance)
        query_second = d_c * moments["second_moment"] * query_std ** 2
        component = d_e * query_second * output_std ** 2
        aggregate = component / num_queries
        gradient_ratio = math.sqrt(d_c * d_e * concept_std ** 2 * query_std ** 2 * moments["derivative_second_moment"] / num_queries)
    return {"concept_preactivation_variance": preactivation_variance, "gelu_gaussian_moments": moments,
            "query_second_moment": query_second, "component_logit_variance": component,
            "component_logit_std": math.sqrt(component),
            "small_score_aggregate_variance": aggregate, "small_score_aggregate_std": math.sqrt(aggregate),
            "small_score_feature_gradient_rms_ratio_vs_dense": gradient_ratio}


def centered_logit_std(logits: Tensor) -> float:
    """Remove the irrelevant per-context logit offset before measuring scale."""
    values = logits.detach().float()
    return float((values - values.mean(dim=-1, keepdim=True)).square().mean().sqrt())


def initialization_variant(head: VocabularyHead, name: str) -> tuple[VocabularyHead, dict]:
    """Clone a head, preserving output rows and random projection directions."""
    if head.normalize_queries or head.normalize_output_rows or head.temperature != 1.0:
        raise ValueError("Diagnostics require the raw-dot, temperature=1 reference.")
    if name not in ("canonical", "fan_in"):
        raise ValueError("variant must be canonical or fan_in")
    cloned = copy.deepcopy(head)
    for parameter in cloned.parameters():
        parameter.grad = None
    concept_std = query_std = 0.02
    with torch.no_grad():
        if name == "fan_in" and isinstance(cloned, MQRHead):
            concept_std, query_std = cloned.d_model ** -0.5, cloned.d_c ** -0.5
            cloned.concept.weight.mul_(concept_std / 0.02)
            cloned.query_projection.weight.mul_(query_std / 0.02)
        elif name == "fan_in" and isinstance(cloned, DirectHead):
            query_std = cloned.d_model ** -0.5
            cloned.query_projection.weight.mul_(query_std / 0.02)
    return cloned, {"variant": name, "concept_weight_std_declared": concept_std,
                    "query_weight_std_declared": query_std, "output_weight_std_declared": 0.02,
                    "query_projection_gain": 1.0, "temperature": 1.0,
                    "direction_preserving_rescale": name == "fan_in"}


@torch.no_grad()
def calibrated_projection_gain(head: VocabularyHead, features: Tensor, target_std: float,
                               *, chunk_positions: int = 32) -> tuple[VocabularyHead, dict]:
    """Clone a raw head and calibrate initialization gain using logits, never labels.

    Gain multiplies the final query projection's initial weights and biases.
    It is not a new trainable gate, loss, normalization, temperature or operator.
    """
    if not isinstance(head, (DirectHead, MQRHead)) or head.normalize_queries or head.normalize_output_rows or head.temperature != 1.0:
        raise ValueError("Gain calibration requires an unnormalized direct/MQR head at temperature 1.")
    if not math.isfinite(target_std) or target_std <= 0:
        raise ValueError("target_std must be finite and positive")
    if type(chunk_positions) is not int or chunk_positions < 1:
        raise ValueError("chunk_positions must be a positive integer")
    cloned = copy.deepcopy(head)
    flat_features = features.reshape(-1, features.shape[-1])
    if not len(flat_features):
        raise ValueError("Calibration requires nonempty features")
    # Cache only component slabs; reductions never materialize logits for more
    # than chunk_positions contexts. The same bank is used at every gain.
    components = [head.component_scores(flat_features[start:start + chunk_positions]).detach().float()
                  for start in range(0, len(flat_features), chunk_positions)]

    def scale(gain: float) -> float:
        square_sum, count = 0.0, 0
        for slab in components:
            logits = torch.logsumexp(slab * gain, dim=-2) - math.log(head.num_queries)
            square_sum += float((logits - logits.mean(-1, keepdim=True)).square().sum())
            count += logits.numel()
        return math.sqrt(square_sum / count)

    baseline_std = scale(1.0)
    upper = 1.0
    while scale(upper) < target_std and upper < 16384:
        upper *= 2.0
    if scale(upper) < target_std:
        raise ValueError("Unable to bracket the requested logit scale")
    lower = 0.0
    for _ in range(24):
        midpoint = (lower + upper) / 2
        if scale(midpoint) < target_std:
            lower = midpoint
        else:
            upper = midpoint
    gain = (lower + upper) / 2
    cloned.query_projection.weight.mul_(gain)
    cloned.query_projection.bias.mul_(gain)
    for parameter in cloned.parameters():
        parameter.grad = None
    return cloned, {"query_projection_gain": gain, "target_centered_logit_std": target_std,
                    "baseline_centered_logit_std": baseline_std,
                    "achieved_centered_logit_std": scale(gain),
                    "chunk_positions": chunk_positions,
                    "projection_sha256_before": initialization_fingerprint(head.query_projection),
                    "projection_sha256_after": initialization_fingerprint(cloned.query_projection),
                    "transformed_parameters": ["query_projection.weight", "query_projection.bias"],
                    "changed_parameters": [name for name, before in head.named_parameters()
                                           if name.startswith("query_projection.") and not torch.equal(before, dict(cloned.named_parameters())[name])],
                    "calibration": "24-step bisection on frozen-feature centered aggregate-logit STD; no targets or losses used"}


@torch.no_grad()
def calibrate_initial_query_gain(head: VocabularyHead, features: Tensor, target_logit_std: float,
                                *, chunk_positions: int = 32) -> dict:
    """Apply the label-free scalar initialization control to the supplied head."""
    cloned, metadata = calibrated_projection_gain(head, features, target_logit_std, chunk_positions=chunk_positions)
    head.query_projection.weight.copy_(cloned.query_projection.weight)
    head.query_projection.bias.copy_(cloned.query_projection.bias)
    return metadata


def _tensor_statistics(values: Tensor) -> dict:
    values = values.detach().float()
    return {"mean": float(values.mean()), "rms": float(values.square().mean().sqrt()),
            "std": float(values.std(correction=0)), "max_abs": float(values.abs().max()),
            "finite": bool(torch.isfinite(values).all())}


def _gradient_group(parameters) -> dict:
    count = grad_count = active = 0
    weight_square = gradient_square = 0.0
    maximum = 0.0
    finite = True
    for parameter in parameters:
        count += parameter.numel()
        flat = parameter.detach().reshape(-1)
        for offset in range(0, flat.numel(), 1_048_576):
            weight_square += float(flat[offset:offset + 1_048_576].double().square().sum())
        if parameter.grad is None:
            continue
        grad_count += parameter.numel()
        grad = parameter.grad.detach().reshape(-1)
        for offset in range(0, grad.numel(), 1_048_576):
            chunk = grad[offset:offset + 1_048_576]
            gradient_square += float(chunk.double().square().sum())
            active += int(torch.count_nonzero(chunk))
            maximum = max(maximum, float(chunk.abs().max()))
            finite = finite and bool(torch.isfinite(chunk).all())
    parameter_rms = math.sqrt(weight_square / count) if count else 0.0
    gradient_rms = math.sqrt(gradient_square / grad_count) if grad_count else 0.0
    return {"parameters": count, "parameters_with_grad": grad_count, "active_gradient_elements": active,
            "parameter_rms": parameter_rms, "gradient_rms": gradient_rms,
            "gradient_l2": math.sqrt(gradient_square), "gradient_max_abs": maximum,
            "active_gradient_rms": math.sqrt(gradient_square / active) if active else 0.0,
            "gradient_to_parameter_rms": gradient_rms / parameter_rms if parameter_rms else None,
            "finite": finite}


def _pair_cosine(values: Tensor) -> float | None:
    if values.shape[-2] < 2:
        return None
    normalized = F.normalize(values.detach().float(), dim=-1)
    cosines = normalized @ normalized.transpose(-1, -2)
    mask = ~torch.eye(values.shape[-2], dtype=torch.bool, device=values.device)
    return float(cosines[..., mask].mean())


def initialization_fingerprint(module) -> str:
    digest = hashlib.sha256()
    for name, values in module.state_dict().items():
        digest.update(f"{name}:{tuple(values.shape)}:{values.dtype}".encode())
        flat = values.detach().cpu().contiguous().reshape(-1)
        for offset in range(0, flat.numel(), 1_048_576):
            digest.update(flat[offset:offset + 1_048_576].numpy().tobytes())
    return digest.hexdigest()


def _head_probe(head: VocabularyHead, features: Tensor, targets: Tensor) -> tuple[dict, Tensor]:
    feature_leaf = features.detach().clone().requires_grad_(True)
    captured = {}
    handles = []

    def capture(name):
        def hook(module, inputs, output):
            output.retain_grad()
            captured[name] = output
        return hook

    if hasattr(head, "concept"):
        handles.append(head.concept.register_forward_hook(capture("concept_preactivation")))
    if hasattr(head, "query_projection"):
        handles.append(head.query_projection.register_forward_hook(capture("projected_queries")))
    try:
        logits = head.score_all(feature_leaf)
        loss = F.cross_entropy(logits.reshape(-1, head.vocab_size), targets.reshape(-1))
        loss.backward()
    finally:
        for handle in handles:
            handle.remove()
    feature_gradient = feature_leaf.grad.detach().clone()
    gradients = {"head_all": _gradient_group(head.parameters()),
                 "head_output": _gradient_group(head.output.parameters())}
    if hasattr(head, "concept"):
        gradients["head_concept"] = _gradient_group(head.concept.parameters())
    if hasattr(head, "query_projection"):
        gradients["head_query_projection"] = _gradient_group(head.query_projection.parameters())
    with torch.no_grad():
        queries = head.make_queries(features)
        components = head.component_scores(features).float()
        log_probabilities = logits.detach().float().log_softmax(-1)
        probabilities = log_probabilities.exp()
        entropy = -(probabilities * log_probabilities).sum(-1)
        residual = logits.detach().float() - components.mean(-2)
        branch_mass = components.logsumexp(-1).softmax(-1)
        gold_indices = targets.unsqueeze(-1).unsqueeze(-1).expand(*targets.shape, head.num_queries, 1)
        gold_responsibility = components.gather(-1, gold_indices).squeeze(-1).softmax(-1)
    query_gradient = (captured["projected_queries"].grad.reshape(*features.shape[:-1], head.num_queries, head.d_e)
                      if "projected_queries" in captured else feature_gradient.unsqueeze(-2))
    result = {"full_ce": float(loss.detach()), "features": _tensor_statistics(features),
              "queries": _tensor_statistics(queries), "query_mean_norm": float(queries.norm(dim=-1).mean()),
              "query_pair_cosine": _pair_cosine(queries), "query_gradient_pair_cosine": _pair_cosine(query_gradient),
              "component_logits": {**_tensor_statistics(components), "centered_std": centered_logit_std(components)},
              "aggregate_logits": {**_tensor_statistics(logits), "centered_std": centered_logit_std(logits)},
              "lse_minus_mean_component": {**_tensor_statistics(residual), "centered_std": centered_logit_std(residual)},
              "output_distribution": {"entropy_nats_mean": float(entropy.mean()),
                                      "uniform_entropy_nats": math.log(head.vocab_size),
                                      "kl_to_uniform_nats_mean": float(math.log(head.vocab_size) - entropy.mean()),
                                      "top_probability_mean": float(probabilities.max(-1).values.mean()),
                                      "top_probability_max": float(probabilities.max())},
              "branch_mass_mean": branch_mass.reshape(-1, head.num_queries).mean(0).tolist(),
              "gold_query_responsibility_mean": gold_responsibility.reshape(-1, head.num_queries).mean(0).tolist(),
              "feature_gradient": _tensor_statistics(feature_gradient), "gradient_groups": gradients,
              "head_parameters": sum(parameter.numel() for parameter in head.parameters())}
    if "concept_preactivation" in captured:
        preactivation = captured["concept_preactivation"].detach()
        result["concept_preactivation"] = _tensor_statistics(preactivation)
        result["concept_gelu"] = _tensor_statistics(F.gelu(preactivation))
        result["concept_preactivation_gradient"] = _tensor_statistics(captured["concept_preactivation"].grad)
    del logits, feature_leaf
    return result, feature_gradient


@torch.enable_grad()
def diagnose_head(head: VocabularyHead, features: Tensor, targets: Tensor) -> dict:
    """JSON-safe exact head/feature probes, clearing temporary gradients afterward.

    Features are detached; no backbone graph is retained or updated. Call before
    training, with clean head gradients. The operation changes no parameters.
    """
    if any(parameter.grad is not None for parameter in head.parameters()):
        raise ValueError("diagnose_head requires clean head gradients before the probe")
    if features.shape[:-1] != targets.shape or features.numel() == 0:
        raise ValueError("Diagnostic targets must match nonempty feature leading dimensions")
    try:
        result, _ = _head_probe(head, features, targets)
        return result
    finally:
        head.zero_grad(set_to_none=True)


def _probe(head: VocabularyHead, features: Tensor, targets: Tensor, backbone: DecoderBackbone, input_ids: Tensor) -> dict:
    result, feature_gradient = _head_probe(head, features, targets)
    backbone.zero_grad(set_to_none=True)
    # Exact chain-rule VJP using the same unchanged decoder and feature gradient.
    graph_features, _ = backbone(input_ids)
    torch.testing.assert_close(graph_features.detach(), features, rtol=0, atol=0)
    graph_features.backward(feature_gradient)
    result["gradient_groups"].update({"backbone_all": _gradient_group(backbone.parameters()),
                                    "backbone_embedding": _gradient_group(backbone.embedding.parameters()),
                                    "backbone_blocks": _gradient_group(backbone.blocks.parameters()),
                                    "backbone_final_norm": _gradient_group(backbone.final_norm.parameters())})
    return result


def diagnose_initialization(config: ExperimentConfig, input_ids: Tensor, targets: Tensor, *, seed: int = 0,
                            experimental: bool = True) -> dict:
    """Run CPU FP32 initialization probes on one shared, unchanged decoder."""
    if input_ids.device.type != "cpu" or targets.device.type != "cpu" or input_ids.shape != targets.shape or input_ids.ndim != 2 or input_ids.numel() < 1:
        raise ValueError("input_ids and targets must be matching nonempty [batch, sequence] CPU tensors")
    if config.head.normalize_queries or config.head.normalize_output_rows or config.head.temperature != 1.0:
        raise ValueError("Initialization diagnosis currently covers the raw-dot temperature=1 reference")
    if config.head.activation != "gelu":
        raise ValueError("Initialization diagnosis requires exact GELU")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        backbone = DecoderBackbone(config.model).eval()
        backbone_identity = initialization_fingerprint(backbone)
        with torch.no_grad():
            features, _ = backbone(input_ids)
        feature_moment = float(features.square().mean())
        reports = {}
        target_std = None
        families = [("dense", "dense", 1), ("direct1", "direct", 1),
                    ("direct4", "direct", config.head.num_queries), ("mqr", "mqr", config.head.num_queries)]
        for family, kind, count in families:
            torch.manual_seed(seed + 1)
            head_config = replace(config.head, kind=kind, num_queries=count)
            canonical = build_head(head_config, config.model.d_model, config.model.vocab_size)
            canonical_identity = initialization_fingerprint(canonical)
            names = ["canonical"] + (["canonical_logit_matched", "fan_in", "fan_in_logit_matched"] if experimental and kind != "dense" else [])
            for name in names:
                clone, initialization = initialization_variant(canonical, "fan_in" if name.startswith("fan_in") else "canonical")
                if name.endswith("_logit_matched"):
                    clone, calibration = calibrated_projection_gain(clone, features, target_std)
                    initialization.update(calibration)
                    initialization["variant"] = name
                    initialization["query_weight_std_declared"] *= calibration["query_projection_gain"]
                report = _probe(clone, features, targets, backbone, input_ids)
                report["initialization"] = initialization
                report["analytical_prediction"] = variance_prediction(
                    kind, config.model.d_model, clone.d_e, config.head.d_c, clone.num_queries,
                    feature_second_moment=feature_moment,
                    concept_std=initialization["concept_weight_std_declared"],
                    query_std=initialization["query_weight_std_declared"],
                )
                reports[f"{family}/{name}"] = report
                if family == "dense":
                    target_std = report["aggregate_logits"]["centered_std"]
                del clone
            if initialization_fingerprint(canonical) != canonical_identity:
                raise RuntimeError("Diagnostic unexpectedly changed canonical head parameters")
            del canonical
        if initialization_fingerprint(backbone) != backbone_identity:
            raise RuntimeError("Diagnostic unexpectedly changed decoder parameters")
        dense = reports["dense/canonical"]
        for report in reports.values():
            report["ratio_vs_dense"] = {
                "centered_aggregate_logit_std": report["aggregate_logits"]["centered_std"] / dense["aggregate_logits"]["centered_std"],
                "feature_gradient_rms": report["feature_gradient"]["rms"] / dense["feature_gradient"]["rms"],
                "head_plus_backbone_gradient_l2": math.hypot(report["gradient_groups"]["head_all"]["gradient_l2"], report["gradient_groups"]["backbone_all"]["gradient_l2"]) / math.hypot(dense["gradient_groups"]["head_all"]["gradient_l2"], dense["gradient_groups"]["backbone_all"]["gradient_l2"]),
            }
        return {"schema_version": 1, "purpose": "initialization mechanics, not trained model quality",
                "feature_source": "one shared untrained decoder, same real input/target tensors for every head",
                "device": "cpu", "precision": "torch.float32", "seed": seed,
                "predicted_positions": input_ids.numel(), "short_context_length": input_ids.shape[1],
                "shape_config": {"model": asdict(config.model), "reference_head": asdict(config.head)},
                "backbone_sha256_before_and_after": backbone_identity,
                "backbone_parameters": sum(parameter.numel() for parameter in backbone.parameters()),
                "canonical_parameters_unchanged": True, "optimizer_updates": 0,
                "loss_reduction": "mean over all supplied predicted positions, full vocabulary CE",
                "reports": reports,
                "analytical_explanation": {
                    "symbols": "d=decoder width, c=concept width, e=query width, M=queries; sigma_C/Q/E are independent zero-mean weight standard deviations; u=feature second moment",
                    "dense_component_variance": "d*u*sigma_E^2",
                    "direct_query_second_moment": "d*u*sigma_Q^2",
                    "mqr_query_second_moment": "c*sigma_Q^2*E[GELU(Z)^2], Z~Normal(0,d*u*sigma_C^2)",
                    "component_variance": "e*sigma_E^2*E[q_coordinate^2]",
                    "small_score_lse": "logmeanexp_j(a_j)=mean_j(a_j)+0.5*population_variance_j(a_j)+higher terms; first-order aggregate variance is component variance/M for independent branches",
                    "small_z_gelu": "GELU(z)=z/2+z^2/sqrt(2*pi)+O(z^4); leading second moment is Var(Z)/4, with quadrature used in numeric predictions",
                    "feature_gradient_ratio_mqr": "sqrt(c*e*sigma_C^2*sigma_Q^2*E[GELU'(Z)^2]/M), versus dense with same sigma_E, near-uniform responsibilities",
                },
                "limitations": [
                    "The decoder is untrained and the context/batch is deliberately small; no prior remote checkpoint is loaded.",
                    "CPU FP32 gradient magnitudes depend on position count and do not reproduce GPU mixed-precision effective-batch norms.",
                    "Small-score independence/mean-field variance predictions are approximations, especially after logit-matching gain.",
                    "Adam normalizes gradients by running moments: a smaller raw gradient norm alone does not imply proportionally smaller optimizer updates or prove the cause of a learning gap.",
                    "Logit gain is calibrated without targets; neither initialization loss nor these diagnostics select a winning variant.",
                    "Use paired-seed, matched-token validation pilots to compare canonical and explicitly named initialization controls before changing a research configuration.",
                ]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--tokens", required=True, type=Path, help="Training split only, .npy token stream")
    parser.add_argument("--positions", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.positions < 1 or args.threads < 1:
        parser.error("positions and threads must be positive")
    torch.set_num_threads(args.threads)
    config = ExperimentConfig.from_json(args.config)
    tokens = np.load(args.tokens, mmap_mode="r", allow_pickle=False)
    if tokens.ndim != 1 or len(tokens) < args.positions + 1 or args.positions > config.model.context_length:
        parser.error("token stream/context does not support the requested diagnostic positions")
    rng = np.random.default_rng(args.seed)
    start = int(rng.integers(0, len(tokens) - args.positions))
    window = np.asarray(tokens[start:start + args.positions + 1], dtype=np.int64).copy()
    tokens._mmap.close()
    report = diagnose_initialization(config, torch.from_numpy(window[:-1]).unsqueeze(0),
                                     torch.from_numpy(window[1:]).unsqueeze(0), seed=args.seed)
    report["input_provenance"] = {"tokens": str(args.tokens.resolve()), "tokens_sha256": file_sha256(args.tokens),
                                   "source_token_start": start, "window_tokens_sha256": hashlib.sha256(window.astype("<i8").tobytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({name: {"logit_std": result["aggregate_logits"]["centered_std"],
                            "feature_gradient_rms": result["feature_gradient"]["rms"],
                            "ratio_vs_dense": result["ratio_vs_dense"],
                            "gain": result["initialization"]["query_projection_gain"]}
                      for name, result in report["reports"].items()}, indent=2))


if __name__ == "__main__":
    main()
