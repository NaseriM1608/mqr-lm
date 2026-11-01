"""Synchronized greedy head and KV-cached decode measurements."""

from __future__ import annotations

import statistics
import time
from typing import Callable

import numpy as np
import torch

from .metrics import coverage_metrics, query_diagnostics
from .retrieval import FaissIPIndex, retrieve_candidates


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(function: Callable, device: torch.device, repeats: int, warmup: int) -> dict:
    for _ in range(warmup):
        function()
    values = []
    for _ in range(repeats):
        _sync(device)
        start = time.perf_counter()
        function()
        _sync(device)
        values.append((time.perf_counter() - start) * 1000)
    return {"p50_ms": statistics.median(values), "p95_ms": float(np.percentile(values, 95)), "raw_ms": values}


@torch.inference_mode()
def benchmark_model(model, prompt: torch.Tensor, *, new_tokens: int = 8, k: int = 32,
                    method: str = "flat", repeats: int = 10, warmup: int = 2,
                    always_include: tuple[int, ...] = (), hnsw_m: int = 32,
                    ef_search: int = 128, ef_construction: int = 200) -> dict:
    """Compare exact scoring and retrieval on identical forced continuations.

    This is a timing diagnostic, not free-generation quality evaluation. Flat
    retrieval is exhaustive; CPU Faiss on a CUDA model includes transfers.
    Decode starts from a completed prefill: N output predictions require N head
    calls and N-1 cached backbone calls. Prefill is timed separately and in the
    combined measurement. This convention is reported with the measurements.
    """
    if repeats < 1 or warmup < 0 or new_tokens < 1:
        raise ValueError("repeats/new_tokens must be positive and warmup nonnegative.")
    if prompt.ndim != 2 or prompt.shape[1] + new_tokens > model.config.model.context_length:
        raise ValueError("Prompt and generation must fit the configured context.")
    device = next(model.parameters()).device
    prompt = prompt.to(device)
    previous_mode = model.training
    model.eval()
    index = None
    build_seconds = 0.0
    try:
        if method in ("faiss", "hnsw"):
            # Exclude earlier asynchronous work and include construction's transfers.
            _sync(device)
            build_start = time.perf_counter()
            index = FaissIPIndex(model.head, "flat" if method == "faiss" else "hnsw",
                                 hnsw_m=hnsw_m, ef_search=ef_search, ef_construction=ef_construction)
            _sync(device)
            build_seconds = time.perf_counter() - build_start
        features, prefix_cache = model.backbone(prompt, use_cache=True)
        last_features = features[:, -1]
        exact = model.head.score_all(last_features).float()
        candidates = retrieve_candidates(model.head, last_features, k, method, index, always_include)
        coverage = coverage_metrics(exact, candidates.ids, top_k=min(50, model.head.vocab_size))
        diagnostics = query_diagnostics(model.head, last_features)

        def choose(f, retrieval: bool):
            if not retrieval:
                return model.head.score_all(f).argmax(-1, keepdim=True)
            c = retrieve_candidates(model.head, f, k, method, index, always_include)
            return c.ids.gather(-1, c.scores.argmax(-1, keepdim=True))

        # Exact model continuation fixes backbone states across timing modes.
        continuation = []
        f, cache = last_features, prefix_cache
        for position in range(new_tokens):
            token = choose(f, False)
            continuation.append(token)
            if position + 1 < new_tokens:
                outputs, cache = model.backbone(token, past_key_values=cache, use_cache=True)
                f = outputs[:, -1]

        def decode(retrieval: bool, prefill: bool):
            if prefill:
                output, cache = model.backbone(prompt, use_cache=True)
                f = output[:, -1]
            else:
                f, cache = last_features, prefix_cache
            for position, forced in enumerate(continuation):
                choose(f, retrieval)
                if position + 1 < new_tokens:
                    output, cache = model.backbone(forced, past_key_values=cache, use_cache=True)
                    f = output[:, -1]

        measurements = {}
        for name, retrieval in (("exact", False), ("retrieval", True)):
            head = _measure(lambda: choose(last_features, retrieval), device, repeats, warmup)
            decode_result = _measure(lambda: decode(retrieval, False), device, repeats, warmup)
            end_to_end = _measure(lambda: decode(retrieval, True), device, repeats, warmup)
            measurements[name] = {"head": head, "decode": decode_result, "prefill_and_decode": end_to_end,
                                  "decode_ms_per_token": decode_result["p50_ms"] / new_tokens,
                                  "decode_tokens_s": prompt.shape[0] * new_tokens * 1000 / decode_result["p50_ms"]}
        prefill = _measure(lambda: model.backbone(prompt, use_cache=True), device, repeats, warmup)
        return {"scope": "forced_continuation_greedy_timing_not_generation_quality", "device": str(device),
                "precision": str(next(model.parameters()).dtype), "cpu_threads": torch.get_num_threads(),
                "batch_size": prompt.shape[0], "prompt_tokens": prompt.shape[1], "new_tokens": new_tokens,
                "vocab_size": model.head.vocab_size, "M": model.head.num_queries, "k": k,
                "ann_method": method, "index_build_seconds": build_seconds,
                "retrieval_execution": {
                    "search_device": "cpu" if index is not None else str(device),
                    "index_dtype": "torch.float32" if index is not None else None,
                    "search_query_dtype": "torch.float32" if index is not None else str(model.head.output.weight.dtype),
                    "rerank_device": str(device),
                    "rerank_component_dtype": str(model.head.output.weight.dtype),
                    "exact_aggregate_dtype": str(exact.dtype),
                    "candidate_aggregate_dtype": str(candidates.scores.dtype),
                    "hybrid_cpu_faiss": index is not None and device.type != "cpu",
                    "device_transfers_included": index is not None and device.type != "cpu",
                    "staleness_check": "parameter_identity_storage_version_geometry" if index is not None else None,
                    "full_table_fingerprint_per_query": False,
                },
                "decode_convention": {
                    "initial_state": "completed_prefill_with_last_features_and_kv_cache",
                    "head_predictions_per_sequence": new_tokens,
                    "cached_backbone_steps_per_sequence": new_tokens - 1,
                    "decode_includes_prefill": False,
                    "prefill_and_decode_includes_prefill": True,
                    "forced_continuation_source": "exact_aggregate_greedy",
                },
                "index_parameters": {"hnsw_m": hnsw_m, "ef_search": ef_search,
                                     "ef_construction": ef_construction} if index is not None else None,
                "coverage_at_prompt": {name: value.float().mean().item() for name, value in coverage.items()},
                "query_diagnostics_at_prompt": {
                    name: (value.float().reshape(-1, model.head.num_queries).mean(0).tolist()
                           if name in ("query_norms", "branch_mass") else value.float().mean().item())
                    for name, value in diagnostics.items()},
                "prefill": prefill, "measurements": measurements}
    finally:
        model.train(previous_mode)
