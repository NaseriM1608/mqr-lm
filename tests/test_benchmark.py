import pytest
import torch

import mqr.benchmark as benchmark_module
from mqr.benchmark import benchmark_model
from mqr.config import ExperimentConfig, HeadConfig, ModelConfig
from mqr.model import LanguageModel


def small_model():
    return LanguageModel(ExperimentConfig(
        model=ModelConfig(vocab_size=11, layers=1, d_model=16, attention_heads=2,
                          ffn_dim=32, context_length=16),
        head=HeadConfig(kind="mqr", d_e=8, d_c=10, num_queries=2),
    ))


@pytest.mark.parametrize("training", [True, False])
def test_benchmark_runs_and_reports_decode_convention(training):
    model = small_model().train(training)
    prompt = torch.tensor([[1, 2, 3], [3, 2, 1]])
    result = benchmark_model(model, prompt, new_tokens=3, k=11, repeats=1, warmup=0)
    assert model.training is training
    assert result["scope"] == "forced_continuation_greedy_timing_not_generation_quality"
    assert result["decode_convention"]["head_predictions_per_sequence"] == 3
    assert result["decode_convention"]["cached_backbone_steps_per_sequence"] == 2
    assert result["decode_convention"]["decode_includes_prefill"] is False
    assert result["retrieval_execution"]["search_device"] == "cpu"
    assert result["retrieval_execution"]["index_dtype"] is None
    assert result["index_build_seconds"] == 0.0
    assert result["coverage_at_prompt"]["retained_mass"] == pytest.approx(1.)
    assert result["coverage_at_prompt"]["top1_agreement"] == 1.
    for mode in ("exact", "retrieval"):
        measurements = result["measurements"][mode]
        assert measurements["head"]["p50_ms"] > 0
        assert measurements["decode"]["p50_ms"] > 0
        assert measurements["prefill_and_decode"]["p50_ms"] > 0
        assert len(measurements["decode"]["raw_ms"]) == 1


@pytest.mark.parametrize("training", [True, False])
def test_index_build_failure_restores_model_mode(monkeypatch, training):
    model = small_model().train(training)

    def fail_build(*args, **kwargs):
        assert not model.training
        raise RuntimeError("index build failed")

    monkeypatch.setattr(benchmark_module, "FaissIPIndex", fail_build)
    with pytest.raises(RuntimeError, match="index build failed"):
        benchmark_model(model, torch.tensor([[1, 2]]), method="faiss", new_tokens=1,
                        repeats=1, warmup=0)
    assert model.training is training


@pytest.mark.parametrize("method", ["faiss", "hnsw"])
def test_index_all_candidates_matches_exact_coverage(method):
    pytest.importorskip("faiss")
    model = small_model().eval()
    result = benchmark_model(model, torch.tensor([[1, 2, 3]]), method=method,
                             new_tokens=2, k=11, repeats=1, warmup=0)
    coverage = result["coverage_at_prompt"]
    assert coverage["candidate_count"] == 11
    assert coverage["retained_mass"] == pytest.approx(1.)
    assert coverage["topk_recall"] == 1.
    assert coverage["top1_agreement"] == 1.
    assert result["index_build_seconds"] > 0
    execution = result["retrieval_execution"]
    assert execution["index_dtype"] == "torch.float32"
    assert execution["search_query_dtype"] == "torch.float32"
    assert execution["rerank_component_dtype"] == "torch.float32"
    assert execution["candidate_aggregate_dtype"] == "torch.float32"
    assert execution["hybrid_cpu_faiss"] is False
    assert execution["device_transfers_included"] is False
    assert execution["full_table_fingerprint_per_query"] is False


def test_index_build_has_sync_before_and_after(monkeypatch):
    model = small_model()
    events = []
    original_index = benchmark_module.FaissIPIndex

    def build(*args, **kwargs):
        events.append("build")
        return original_index(*args, **kwargs)

    pytest.importorskip("faiss")
    monkeypatch.setattr(benchmark_module, "_sync", lambda device: events.append("sync"))
    monkeypatch.setattr(benchmark_module, "FaissIPIndex", build)
    benchmark_model(model, torch.tensor([[1, 2]]), method="faiss", new_tokens=1,
                    k=11, repeats=1, warmup=0)
    assert events[:3] == ["sync", "build", "sync"]
