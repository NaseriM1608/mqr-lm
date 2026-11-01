import pytest
import torch

import mqr.model as model_module
from mqr.config import ExperimentConfig, HeadConfig, ModelConfig
from mqr.model import LanguageModel
from mqr.retrieval import CandidateBatch


def small_model(kind="mqr"):
    return LanguageModel(ExperimentConfig(
        model=ModelConfig(vocab_size=11, layers=1, d_model=16, attention_heads=2,
                          ffn_dim=32, context_length=16),
        head=HeadConfig(kind=kind, d_e=8, d_c=10, num_queries=2),
    ))


@pytest.mark.parametrize("kind", ["dense", "direct", "mqr"])
@pytest.mark.parametrize("method", ["flat", "faiss", "hnsw"])
def test_all_candidates_greedy_generation_matches_exact(kind, method):
    if method != "flat":
        pytest.importorskip("faiss")
    torch.manual_seed(22)
    model = small_model(kind)
    prompt = torch.tensor([[1, 2, 3], [3, 2, 1]])
    exact = model.generate(prompt, 4, 0.0)
    retrieved = model.generate(prompt, 4, 0.0, method=method, k=11)
    torch.testing.assert_close(retrieved, exact)
    assert model.training


@pytest.mark.parametrize("top_k", [None, 3, 11])
def test_all_candidates_sampling_matches_exact_with_shared_seed(top_k):
    model = small_model()
    prompt = torch.tensor([[1, 2, 3], [3, 2, 1]])
    torch.manual_seed(36)
    exact = model.generate(prompt, 4, 0.8, top_k)
    torch.manual_seed(36)
    retrieved = model.generate(prompt, 4, 0.8, top_k, method="flat", k=11)
    torch.testing.assert_close(retrieved, exact)


@pytest.mark.parametrize("method", ["faiss", "hnsw"])
def test_index_is_built_once_and_reused_for_all_steps(monkeypatch, method):
    pytest.importorskip("faiss")
    model = small_model().eval()
    original_index = model_module.FaissIPIndex
    original_retrieve = model_module.retrieve_candidates
    built = []
    searched = []

    def build(head, algorithm):
        index = original_index(head, algorithm)
        built.append(index)
        return index

    def retrieve(head, features, k, algorithm, index, always_include):
        searched.append(index)
        return original_retrieve(head, features, k, algorithm, index, always_include)

    monkeypatch.setattr(model_module, "FaissIPIndex", build)
    monkeypatch.setattr(model_module, "retrieve_candidates", retrieve)
    model.generate(torch.tensor([[1, 2]]), 4, 0.0, method=method, k=4)
    assert len(built) == 1
    assert len(searched) == 4
    assert all(index is built[0] for index in searched)
    assert not model.training


@pytest.mark.parametrize("training", [True, False])
def test_index_failure_restores_mode(monkeypatch, training):
    model = small_model().train(training)

    def build(*args, **kwargs):
        assert not model.training
        raise RuntimeError("index failed")

    monkeypatch.setattr(model_module, "FaissIPIndex", build)
    with pytest.raises(RuntimeError, match="index failed"):
        model.generate(torch.tensor([[1, 2]]), 2, method="faiss")
    assert model.training is training


def test_retrieval_failure_restores_mode(monkeypatch):
    model = small_model().train()

    def retrieve(*args, **kwargs):
        raise RuntimeError("retrieval failed")

    monkeypatch.setattr(model_module, "retrieve_candidates", retrieve)
    with pytest.raises(RuntimeError, match="retrieval failed"):
        model.generate(torch.tensor([[1, 2]]), 2, method="flat")
    assert model.training


@pytest.mark.parametrize("method,k", [("bad", 1), (None, 1), ("flat", 0), ("exact", -1), ("hnsw", True)])
def test_invalid_retrieval_options_are_rejected(method, k):
    with pytest.raises(ValueError):
        small_model().generate(torch.tensor([[1, 2]]), 2, method=method, k=k)


def test_eos_is_included_scored_normally_and_stops_generation(monkeypatch):
    model = small_model()
    calls = []

    def retrieve(head, features, k, method, index, always_include):
        calls.append(always_include)
        assert always_include == (10,)
        # EOS can win by its score; generation must not force it over higher scores.
        return CandidateBatch(torch.tensor([[3, 10]]), torch.tensor([[0., 1.]]), torch.tensor([2]))

    monkeypatch.setattr(model_module, "retrieve_candidates", retrieve)
    result = model.generate(torch.tensor([[1, 2]]), 4, 0.0, eos_token_id=10, method="flat", k=1)
    assert result.tolist() == [[1, 2, 10]]
    assert len(calls) == 1


def test_eos_candidate_is_not_automatically_preferred(monkeypatch):
    model = small_model()

    def retrieve(head, features, k, method, index, always_include):
        assert always_include == (10,)
        return CandidateBatch(torch.tensor([[3, 10]]), torch.tensor([[1., 0.]]), torch.tensor([2]))

    monkeypatch.setattr(model_module, "retrieve_candidates", retrieve)
    result = model.generate(torch.tensor([[1, 2]]), 2, 0.0, eos_token_id=10, method="flat", k=1)
    assert result.tolist() == [[1, 2, 3, 3]]


def test_top_k_larger_than_candidate_union_remains_valid():
    model = small_model()
    result = model.generate(torch.tensor([[1, 2]]), 3, 0.8, top_k=11, method="flat", k=1)
    assert result.shape == (1, 5)
    assert ((result >= 0) & (result < 11)).all()
